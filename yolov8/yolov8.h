#ifndef YOLOV8_INFERENCE_H_
#define YOLOV8_INFERENCE_H_

#include <iostream>
#include <string>
#include <vector>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <openvino/openvino.hpp>

namespace yolov8 {
struct Detection {
	short class_id;
	float confidence;
	cv::Rect box;
};

class Inference {
 public:
	Inference() {}
	Inference(const std::string &model_path, const short &width, const short &height);

	std::vector<Detection> RunInference(const cv::Mat &frame);

 private:
	void InitialModel(const std::string &model_path);
	void Preprocessing(const cv::Mat &frame);
	void PostProcessing(const float *detections, const ov::Shape &output_shape);
	cv::Rect GetBoundingBox(const cv::Rect &src);

	cv::Mat resized_frame_;
	cv::Point2f factor_;
	cv::Size2f model_input_shape_;

	ov::Tensor input_tensor_;
	ov::InferRequest inference_request_;
	ov::CompiledModel compiled_model_;

	std::vector<Detection> detections_;

	float model_score_threshold_;
	float model_NMS_threshold_;
};
} // namespace yolov8

#endif // YOLOV8_INFERENCE_H_
