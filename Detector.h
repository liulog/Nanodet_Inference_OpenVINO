#ifndef _NANODET_OPENVINO_H_
#define _NANODET_OPENVINO_H_

#include <string>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <inference_engine.hpp>


// 检测头信息
typedef struct HeadInfo{ std::string cls_layer; std::string dis_layer; int stride;} HeadInfo;

// 中心点prior
struct CenterPrior{ int x; int y; int stride;};

// 检验框信息
typedef struct BoxInfo{ float x1; float y1; float x2; float y2; float score; int label;} BoxInfo;

// 四点信息
typedef struct PtsInfo{ float x1; float y1; float x2; float y2; float x3; float y3; float x4; float y4; float score; int label;} PtsInfo;

// resize后图片在模型在图片中的roi
struct object_rect { int x; int y; int width; int height; };

// 检测类
class NanoDet
{
public:
    NanoDet(const char* param);
    ~NanoDet();

    // OpenVINO推理相关
    InferenceEngine::ExecutableNetwork network_;
    InferenceEngine::InferRequest infer_request_;
    
    // 模型的输入大小
    int input_size[2] = {416, 416};  
    // 检测类别数
    int num_class = 36;
    // reg_max 根据训练确定，默认7即可
    int reg_max = 7; 
    // strides 步长
    std::vector<int> strides = { 8, 16, 32, 64 };  

    /**
     * detect: 检测函数（输入输出都为原图）
     *      image: 输入的图像（原图输入，内部进行resize）
     *      score_threshold: 检测得分阈值
     *      nms_threshold: nms后处理阈值
     *      bbox_dets: 检验框结果存放集（包括score和label）
     *      pts_dets: 四点检测结果集（包括score和label）
     */
    void detect(cv::Mat &image, float score_threshold, float nms_threshold, std::vector<BoxInfo>& bbox_dets, std::vector<PtsInfo>& pts_dets);

    /**
     * draw_bboxes  绘制出检测框以及四点的位置（输入原图以及bboxes和points） 注意该函数会自动复制一份图像，不需要提前拷贝图像
     *      bgr: 输入的原始图像 
     *      bboxes: bboxes存放vector
     *      points: points存放vector
     *      effect_roi: 相关区域存放
     */
    void draw_bboxes(const cv::Mat& bgr, const std::vector<BoxInfo>& bboxes, const std::vector<PtsInfo>& points);


private:
    // 预处理
    void preprocess(cv::Mat& image, InferenceEngine::Blob::Ptr& blob);
    // 输出解码
    void decode_infer(const float*& pred, std::vector<CenterPrior>& center_priors, float threshold, std::vector<std::vector<BoxInfo>>& bbox_results, std::vector<std::vector<PtsInfo>>& pts_results);
    
    // 相关转换函数
    BoxInfo disPred2Bbox(const float*& dfl_det, int label, float score, int x, int y, int stride);
    PtsInfo disPred2Pts(const float*& dfl_det, int label, float score, int x, int y, int stride);

    // nms处理
    static void nms(std::vector<BoxInfo>& box_result, std::vector<PtsInfo>& pts_result, float nms_threshold);

    // resize处理
    int resize_uniform(cv::Mat& src, cv::Mat& dst, cv::Size dst_size, object_rect& effect_area);

    object_rect effect_roi;
    std::string input_name_ = "data";
    std::string output_name_ = "output";
};

#endif //_NANODE_TOPENVINO_H_