#include "yolov8.h"

#include <memory>

namespace yolov8 {
Inference::Inference(const std::string &model_path, const short &width, const short &height) {
	model_score_threshold_ = 0.48;
	model_NMS_threshold_ = 0.48;
	model_input_shape_ = cv::Size2f(height, width);
	InitialModel(model_path);
}

void Inference::InitialModel(const std::string &model_path) {
	ov::Core core;
	std::shared_ptr<ov::Model> model = core.read_model(model_path);
	ov::preprocess::PrePostProcessor ppp = ov::preprocess::PrePostProcessor(model);

  ppp.input().tensor().set_element_type(ov::element::u8).set_layout("NHWC").set_color_format(ov::preprocess::ColorFormat::BGR);
  ppp.input().preprocess().convert_element_type(ov::element::f32).convert_color(ov::preprocess::ColorFormat::RGB).scale({ 255, 255, 255 });
	ppp.input().model().set_layout("NCHW");
  ppp.output().tensor().set_element_type(ov::element::f32);

  model = ppp.build();
	compiled_model_ = core.compile_model(model, "CPU");
	inference_request_ = compiled_model_.create_infer_request();
}

std::vector<Detection> Inference::RunInference(const cv::Mat &frame) {
	Preprocessing(frame);
	inference_request_.infer();

	const ov::Tensor &output_tensor = inference_request_.get_output_tensor();
	ov::Shape output_shape = output_tensor.get_shape();
	float *detections = output_tensor.data<float>();

	PostProcessing(detections, output_shape);

	return detections_;
}

void Inference::Preprocessing(const cv::Mat &frame) {
	cv::resize(frame, resized_frame_, model_input_shape_, 0, 0, cv::INTER_AREA);

	factor_.x = static_cast<float>(frame.cols / model_input_shape_.width);
	factor_.y = static_cast<float>(frame.rows / model_input_shape_.height);

	float *input_data = (float *)resized_frame_.data;
	input_tensor_ = ov::Tensor(compiled_model_.input().get_element_type(), compiled_model_.input().get_shape(), input_data);
	inference_request_.set_input_tensor(input_tensor_);
}

void Inference::PostProcessing(const float *detections, const ov::Shape &output_shape) {
	std::vector<int> class_list;
	std::vector<float> confidence_list;
	std::vector<cv::Rect> box_list;

	const int output_rows = output_shape[1];
	const int output_cols = output_shape[2];
	const cv::Mat detection_outputs(output_rows, output_cols, CV_32F, (float *)detections);

	for (int i = 0; i < detection_outputs.cols; ++i) {
		const cv::Mat classes_scores = detection_outputs.col(i).rowRange(4, detection_outputs.rows);

		cv::Point class_id;
		double score;

		cv::minMaxLoc(classes_scores, nullptr, &score, nullptr, &class_id);

		if (score > model_score_threshold_) {
			class_list.push_back(class_id.y);
			confidence_list.push_back(score);

			const float x = detection_outputs.at<float>(0, i);
			const float y = detection_outputs.at<float>(1, i);
			const float w = detection_outputs.at<float>(2, i);
			const float h = detection_outputs.at<float>(3, i);

			cv::Rect box;

			box.x = static_cast<int>(x);
			box.y = static_cast<int>(y);
			box.width = static_cast<int>(w);
			box.height = static_cast<int>(h);

			box_list.push_back(box);
		}
	}

	std::vector<int> NMS_result;
	cv::dnn::NMSBoxes(box_list, confidence_list, model_score_threshold_, model_NMS_threshold_, NMS_result);

	detections_.clear();

	for (int i = 0; i < NMS_result.size(); ++i) {
		Detection result;
		int id = NMS_result[i];

		result.class_id = class_list[id];
		result.confidence = confidence_list[id];
		result.box = GetBoundingBox(box_list[id]);

		detections_.push_back(result);
	}
}

cv::Rect Inference::GetBoundingBox(const cv::Rect &src) {
	cv::Rect box = src;

	box.x = (box.x - 0.5 * box.width) * factor_.x;
	box.y = (box.y - 0.5 * box.height) * factor_.y;
	box.width *= factor_.x;
	box.height *= factor_.y;
	
	return box;
}
} // namespace yolov8
