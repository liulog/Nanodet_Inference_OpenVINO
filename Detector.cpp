#include "Detector.h"


// COCO数据集用来给不同类别用不同颜色的框
const int color_list[80][3] =
{
    //{255 ,255 ,255}, //bg
    {216 , 82 , 24},
    {236 ,176 , 31},
    {125 , 46 ,141},
    {118 ,171 , 47},
    { 76 ,189 ,237},
    {238 , 19 , 46},
    { 76 , 76 , 76},
    {153 ,153 ,153},
    {255 ,  0 ,  0},
    {255 ,127 ,  0},
    {190 ,190 ,  0},
    {  0 ,255 ,  0},
    {  0 ,  0 ,255},
    {170 ,  0 ,255},
    { 84 , 84 ,  0},
    { 84 ,170 ,  0},
    { 84 ,255 ,  0},
    {170 , 84 ,  0},
    {170 ,170 ,  0},
    {170 ,255 ,  0},
    {255 , 84 ,  0},
    {255 ,170 ,  0},
    {255 ,255 ,  0},
    {  0 , 84 ,127},
    {  0 ,170 ,127},
    {  0 ,255 ,127},
    { 84 ,  0 ,127},
    { 84 , 84 ,127},
    { 84 ,170 ,127},
    { 84 ,255 ,127},
    {170 ,  0 ,127},
    {170 , 84 ,127},
    {170 ,170 ,127},
    {170 ,255 ,127},
    {255 ,  0 ,127},
    {255 , 84 ,127},
    {255 ,170 ,127},
    {255 ,255 ,127},
    {  0 , 84 ,255},
    {  0 ,170 ,255},
    {  0 ,255 ,255},
    { 84 ,  0 ,255},
    { 84 , 84 ,255},
    { 84 ,170 ,255},
    { 84 ,255 ,255},
    {170 ,  0 ,255},
    {170 , 84 ,255},
    {170 ,170 ,255},
    {170 ,255 ,255},
    {255 ,  0 ,255},
    {255 , 84 ,255},
    {255 ,170 ,255},
    { 42 ,  0 ,  0},
    { 84 ,  0 ,  0},
    {127 ,  0 ,  0},
    {170 ,  0 ,  0},
    {212 ,  0 ,  0},
    {255 ,  0 ,  0},
    {  0 , 42 ,  0},
    {  0 , 84 ,  0},
    {  0 ,127 ,  0},
    {  0 ,170 ,  0},
    {  0 ,212 ,  0},
    {  0 ,255 ,  0},
    {  0 ,  0 , 42},
    {  0 ,  0 , 84},
    {  0 ,  0 ,127},
    {  0 ,  0 ,170},
    {  0 ,  0 ,212},
    {  0 ,  0 ,255},
    {  0 ,  0 ,  0},
    { 36 , 36 , 36},
    { 72 , 72 , 72},
    {109 ,109 ,109},
    {145 ,145 ,145},
    {182 ,182 ,182},
    {218 ,218 ,218},
    {  0 ,113 ,188},
    { 80 ,182 ,188},
    {127 ,127 ,  0},
};


// 快速指数计算
inline float fast_exp(float x)
{
    union {
        uint32_t i;
        float f;
    } v{};
    v.i = (1 << 23) * (1.4426950409 * x + 126.93490512f);
    return v.f;
}

// sigmoid函数
inline float sigmoid(float x)
{
    return 1.0f / (1.0f + fast_exp(-x));
}

// softmax处理
template<typename _Tp>
int activation_function_softmax(const _Tp* src, _Tp* dst, int length)
{
    const _Tp alpha = *std::max_element(src, src + length);
    _Tp denominator{ 0 };

    for (int i = 0; i < length; ++i)
    {
        dst[i] = fast_exp(src[i] - alpha);
        denominator += dst[i];
    }

    for (int i = 0; i < length; ++i)
    {
        dst[i] /= denominator;
    }

    return 0;
}

// 生成中心点 input_height 高度， input_width 宽度， strides 步长， center_priors 中心点（注意要 x 步长）
static void generate_grid_center_priors(const int input_height, const int input_width, std::vector<int>& strides, std::vector<CenterPrior>& center_priors)
{
    // 遍历不同步长
    for (int i = 0; i < (int)strides.size(); i++)
    {
        int stride = strides[i];
        int feat_w = ceil((float)input_width / stride);
        int feat_h = ceil((float)input_height / stride);
        for (int y = 0; y < feat_h; y++)
        {
            for (int x = 0; x < feat_w; x++)
            {
                CenterPrior ct;                     // 中心点
                ct.x = x;                           // 1，2，3，连续的，需要 x 步长 ， 才可以得到图片中的真实坐标
                ct.y = y;
                ct.stride = stride;
                center_priors.push_back(ct);
            }
        }
    }
}

// 初始化推理：初始化工作
NanoDet::NanoDet(const char* model_path)
{
    // init
    InferenceEngine::Core ie;
    InferenceEngine::CNNNetwork model = ie.ReadNetwork(model_path);     // 指定模型路径
    
    // prepare input settings
    InferenceEngine::InputsDataMap inputs_map(model.getInputsInfo());
    input_name_ = inputs_map.begin()->first;
    InferenceEngine::InputInfo::Ptr input_info = inputs_map.begin()->second;
    // input_info->setPrecision(InferenceEngine::Precision::FP32);
    // input_info->setLayout(InferenceEngine::Layout::NCHW);
    
    //prepare output settings
    InferenceEngine::OutputsDataMap outputs_map(model.getOutputsInfo());
    for (auto &output_info : outputs_map)
    {
        //std::cout<< "Output:" << output_info.first <<std::endl;
        output_info.second->setPrecision(InferenceEngine::Precision::FP32);
    }
    
    //get network
    network_ = ie.LoadNetwork(model, "CPU");      // 使用CPU推理
    infer_request_ = network_.CreateInferRequest();
}

NanoDet::~NanoDet(){ }

// 预处理
void NanoDet::preprocess(cv::Mat& image, InferenceEngine::Blob::Ptr& blob)
{
    int img_w = image.cols;
    int img_h = image.rows;
    int channels = 3;

    InferenceEngine::MemoryBlob::Ptr mblob = InferenceEngine::as<InferenceEngine::MemoryBlob>(blob);
    if (!mblob)
    {
        THROW_IE_EXCEPTION << "We expect blob to be inherited from MemoryBlob in matU8ToBlob, "
            << "but by fact we were not able to cast inputBlob to MemoryBlob";
    }
    
    // locked memory holder should be alive all time while access to its buffer happens
    auto mblobHolder = mblob->wmap();

    float *blob_data = mblobHolder.as<float *>();

    for (size_t c = 0; c < channels; c++)
    {
        for (size_t  h = 0; h < img_h; h++)
        {
            for (size_t w = 0; w < img_w; w++)
            {
                blob_data[c * img_w * img_h + h * img_w + w] =
                    (float)image.at<cv::Vec3b>(h, w)[c];
            }
        }
    }
}

// 检测 input图片， score_threshold得分阈值， nms_threshold后处理阈值
void NanoDet::detect(cv::Mat &raw_image, float score_threshold, float nms_threshold, std::vector<BoxInfo>& bbox_dets, std::vector<PtsInfo>& pts_dets)
{
    cv::Mat image;                        // 图片大小进行resize

    // 因为场上的effect_roi肯定是固定的，方便起见，还是集成到Detector模块
    resize_uniform(raw_image, image, cv::Size(input_size[1], input_size[0]), effect_roi); // 这里将resize集成到了detect函数中，每次确定对应图片的effect_roi 
    
    // std::cout<<image.cols<<" "<<image.rows<<std::endl;
    // std::cout<<raw_image.cols<<" "<<raw_image.rows<<std::endl;
    // std::cout<<effect_roi.x<<" "<<effect_roi.y<<" "<<effect_roi.height<<" "<<effect_roi.width<<std::endl;
    
    int src_w = raw_image.cols;
    int src_h = raw_image.rows;
    // 目标的尺寸大小
    int dst_w = effect_roi.width;
    int dst_h = effect_roi.height;
    // 确定一下宽度比，高度比，这样可以
    float width_ratio = (float)src_w / (float)dst_w;    // 1920 / 320
    float height_ratio = (float)src_h / (float)dst_h;   // 1080 / (320 * 1080 / 1920)
    // std::cout<<width_ratio<<" "<<height_ratio<<std::endl;
    

    InferenceEngine::Blob::Ptr input_blob = infer_request_.GetBlob(input_name_);
    
    // 预处理 blob就是一种文件读取
    preprocess(image, input_blob);

    // do inference 推理
    infer_request_.Infer();

    // get output 存放结果
    std::vector<std::vector<BoxInfo>> bbox_results;
    std::vector<std::vector<PtsInfo>> pts_results;  // 存放pts_results推理结果

    bbox_results.resize(this->num_class);           // 二维数组 -> 设置为 num_class 行， 固定了行数
    pts_results.resize(this->num_class);
    {
        const InferenceEngine::Blob::Ptr pred_blob = infer_request_.GetBlob(output_name_);

        auto m_pred = InferenceEngine::as<InferenceEngine::MemoryBlob>(pred_blob);
        auto m_pred_holder = m_pred->rmap();
        const float *pred = m_pred_holder.as<const float *>();

        // generate center priors in format of (x, y, stride)
        std::vector<CenterPrior> center_priors;
        generate_grid_center_priors(this->input_size[0], this->input_size[1], this->strides, center_priors);

        // 对推理结果进行解码，pred存放推理的结果
        this->decode_infer(pred, center_priors, score_threshold, bbox_results, pts_results); // 包括了pts_result解码
    }

    // 后处理，nms
    for (int i = 0; i < (int)bbox_results.size(); i++)
    {
        this->nms(bbox_results[i], pts_results[i], nms_threshold);// nms后处理

        for (auto& box : bbox_results[i])
        {
            box.x1 = (box.x1 - effect_roi.x) * width_ratio; // 这里对其进行了处理
            box.y1 = (box.y1 - effect_roi.y) * height_ratio;
            box.x2 = (box.x2 - effect_roi.x) * width_ratio;
            box.y2 = (box.y2 - effect_roi.y) * height_ratio;

            bbox_dets.push_back(box);
        }
        for (auto& pts : pts_results[i])            // 检测框同时输出points相关信息
        {
            pts.x1 = (pts.x1 - effect_roi.x) * width_ratio;
            pts.y1 = (pts.y1 - effect_roi.y) * height_ratio;
            pts.x2 = (pts.x2 - effect_roi.x) * width_ratio;
            pts.y2 = (pts.y2 - effect_roi.y) * height_ratio;
            pts.x3 = (pts.x3 - effect_roi.x) * width_ratio;
            pts.y3 = (pts.y3 - effect_roi.y) * height_ratio;
            pts.x4 = (pts.x4 - effect_roi.x) * width_ratio;
            pts.y4 = (pts.y4 - effect_roi.y) * height_ratio;

            pts_dets.push_back(pts);                    
        }
    }
}

// 解码推理结果 pred是结果地址， center_priors方便bbox， threshold阈值
void NanoDet::decode_infer(const float*& pred, std::vector<CenterPrior>& center_priors, float threshold, std::vector<std::vector<BoxInfo>>& bbox_results, std::vector<std::vector<PtsInfo>>& pts_results)
{
    const int num_points = center_priors.size();             // 中心点的个数
    const int num_channels = num_class + (reg_max + 1) * 12; //通道的个数

    //cv::Mat debug_heatmap = cv::Mat::zeros(feature_h, feature_w, CV_8UC3);
    
    // 输出为 36 + 4*(reg_max+1) + 8*(reg_max+1)
    // 遍历所有中心点，解析出所有的内容
    for (int idx = 0; idx < num_points; idx++)
    {
        const int ct_x = center_priors[idx].x;  
        const int ct_y = center_priors[idx].y;
        const int stride = center_priors[idx].stride;

        float score = 0;
        int cur_label = 0;  // 存放得分最高的类别

        // 先获取这个框这里的类别
        for (int label = 0; label < num_class; label++)     // 遍历所有的类别
        {
            if (pred[idx * num_channels + label] / 10 > score)  // 相当于找出类别的得分最大值，它带有的标签即位该类别的标签
            {
                score = pred[idx * num_channels + label];       // float
                cur_label = label;
            }
        }

        if (score > threshold)  // 得分 大于 阈值
        {
            const float* bbox_pred = pred + idx * num_channels + num_class;     // 找到bbox的起点

            // 根据dis和中心点解析出bbox并将结果存放到
            bbox_results[cur_label].push_back(this->disPred2Bbox(bbox_pred, cur_label, score, ct_x, ct_y, stride));

            const float* pts_pred = pred + idx * num_channels + num_class + 4 * (reg_max + 1);     // TODO 找到bbox的起点

            pts_results[cur_label].push_back(this->disPred2Pts(pts_pred, cur_label, score, ct_x, ct_y, stride));
            //debug_heatmap.at<cv::Vec3b>(row, col)[0] = 255;
            //cv::imshow("debug", debug_heatmap);
        }
    }
}

// center point + dis 转换成 bbox
BoxInfo NanoDet::disPred2Bbox(const float*& dfl_det, int label, float score, int x, int y, int stride)
{
    // 获取原图的中心点的坐标
    float ct_x = x * stride;
    float ct_y = y * stride;
    std::vector<float> dis_pred;
    dis_pred.resize(4); // 存放4个数据
    for (int i = 0; i < 4; i++)
    {
        float dis = 0;
        float* dis_after_sm = new float[reg_max + 1];
        activation_function_softmax(dfl_det + i * (reg_max + 1), dis_after_sm, reg_max + 1);
        // 积分得到最终的结果
        for (int j = 0; j < reg_max + 1; j++)
        {
            dis += j * dis_after_sm[j];
        }
        dis *= stride;
        //std::cout << "dis:" << dis << std::endl;
        dis_pred[i] = dis;
        delete[] dis_after_sm;
    }
    
    // 使用min、max来限制坐标
    // 左上角点
    float xmin = (std::max)(ct_x - dis_pred[0], .0f);   
    float ymin = (std::max)(ct_y - dis_pred[1], .0f);

    // 右下角点
    float xmax = (std::min)(ct_x + dis_pred[2], (float)this->input_size[1]);
    float ymax = (std::min)(ct_y + dis_pred[3], (float)this->input_size[0]);

    // std::cout << xmin << "," << ymin << "," << xmax << "," << xmax << "," << std::endl;
    // 得到bbox信息
    return BoxInfo { xmin, ymin, xmax, ymax, score, label };
}


// 新增 center point + dis 转换成 pts
PtsInfo NanoDet::disPred2Pts(const float*& dfl_det, int label, float score, int x, int y, int stride)
{
    // 获取原图的中心点的坐标
    float ct_x = x * stride;
    float ct_y = y * stride;
    std::vector<float> dis_pred;
    dis_pred.resize(8); // 存放8个数据
    for (int i = 0; i < 8; i++)
    {
        float dis = 0;
        float* dis_after_sm = new float[reg_max + 1];
        activation_function_softmax(dfl_det + i * (reg_max + 1), dis_after_sm, reg_max + 1);
        // 积分得到最终的结果
        for (int j = 0; j < reg_max + 1; j++)
        {
            dis += j * dis_after_sm[j];
        }
        dis *= stride;
        //std::cout << "dis:" << dis << std::endl;
        dis_pred[i] = dis;
        delete[] dis_after_sm;
    }
    
    // 使用min、max来限制坐标
    // 左上角点
    float x1 = (std::max)(ct_x - dis_pred[0], .0f);   
    float y1 = (std::max)(ct_y - dis_pred[1], .0f);

    float x2 = (std::max)(ct_x - dis_pred[2], .0f);   
    float y2 = (std::min)(ct_y + dis_pred[3], (float)this->input_size[0]);

    float x3 = (std::min)(ct_x + dis_pred[4], (float)this->input_size[1]);
    float y3 = (std::min)(ct_x + dis_pred[5], (float)this->input_size[0]);

    // 右下角点
    float x4 = (std::min)(ct_x + dis_pred[6], (float)this->input_size[1]);
    float y4 = (std::max)(ct_x - dis_pred[7], .0f); 

    // std::cout << xmin << "," << ymin << "," << xmax << "," << xmax << "," << std::endl;
    // 得到bbox信息
    return PtsInfo { x1, y1, x2, y2, x3, y3, x4, y4, score, label };
}


// nms, 后处理还是先以bbox比较iou为主           
void NanoDet::nms(std::vector<BoxInfo>& input_boxes, std::vector<PtsInfo>& input_points, float NMS_THRESH)
{
    std::sort(input_boxes.begin(), input_boxes.end(), [](BoxInfo a, BoxInfo b) { return a.score > b.score; });
    std::sort(input_points.begin(), input_points.end(), [](PtsInfo a, PtsInfo b) { return a.score > b.score; });
    std::vector<float> vArea(input_boxes.size());
    for (int i = 0; i < int(input_boxes.size()); ++i)
    {
        vArea[i] = (input_boxes.at(i).x2 - input_boxes.at(i).x1 + 1)
            * (input_boxes.at(i).y2 - input_boxes.at(i).y1 + 1);
    }
    for (int i = 0; i < int(input_boxes.size()); ++i)
    {
        for (int j = i + 1; j < int(input_boxes.size());)
        {
            float xx1 = (std::max)(input_boxes[i].x1, input_boxes[j].x1);
            float yy1 = (std::max)(input_boxes[i].y1, input_boxes[j].y1);
            float xx2 = (std::min)(input_boxes[i].x2, input_boxes[j].x2);
            float yy2 = (std::min)(input_boxes[i].y2, input_boxes[j].y2);
            float w = (std::max)(float(0), xx2 - xx1 + 1);
            float h = (std::max)(float(0), yy2 - yy1 + 1);
            float inter = w * h;
            float ovr = inter / (vArea[i] + vArea[j] - inter);
            if (ovr >= NMS_THRESH)                              // 借助bbox的iou同时也对points进行筛选
            {
                input_boxes.erase(input_boxes.begin() + j);
                input_points.erase(input_points.begin() + j);
                vArea.erase(vArea.begin() + j);
            }
            else
            {
                j++;
            }
        }
    }
}

// resize_uniform调用
int NanoDet::resize_uniform(cv::Mat& src, cv::Mat& dst, cv::Size dst_size, object_rect& effect_area)
{
    int w = src.cols;               // 1920
    int h = src.rows;               // 1080
    int dst_w = dst_size.width;     // 320
    int dst_h = dst_size.height;    // 320
    //std::cout << "src: (" << h << ", " << w << ")" << std::endl;
    dst = cv::Mat(cv::Size(dst_w, dst_h), CV_8UC3, cv::Scalar(0));

    float ratio_src = w * 1.0 / h;          // 原图宽高比 w/h < 1
    float ratio_dst = dst_w * 1.0 / dst_h;  // 目标图宽高比 w/h = 1

    int tmp_w = 0;
    int tmp_h = 0;
    if (ratio_src > ratio_dst) {            
        tmp_w = dst_w;                          // tmp_w = 320
        tmp_h = floor((dst_w * 1.0 / w) * h);   // tmp_h = 320 * 1080 / 1920
    }
    else if (ratio_src < ratio_dst) {
        tmp_h = dst_h;
        tmp_w = floor((dst_h * 1.0 / h) * w);
    }
    else {
        cv::resize(src, dst, dst_size);
        effect_area.x = 0;
        effect_area.y = 0;
        effect_area.width = dst_w;
        effect_area.height = dst_h;
        return 0;
    }

    cv::Mat tmp;                                
    cv::resize(src, tmp, cv::Size(tmp_w, tmp_h));   // 将原图长宽等比例缩放

    if (tmp_w != dst_w) {                           // 如果宽度没充满 tmp_w = dst_w = 320
        int index_w = floor((dst_w - tmp_w) / 2.0);

        for (int i = 0; i < dst_h; i++) {
            memcpy(dst.data + i * dst_w * 3 + index_w * 3, tmp.data + i * tmp_w * 3, tmp_w * 3);
        }
        effect_area.x = index_w;
        effect_area.y = 0;
        effect_area.width = tmp_w;
        effect_area.height = tmp_h;
    }
    else if (tmp_h != dst_h) {                      // 高度没充满, tmp_h < dst_w = 320
        int index_h = floor((dst_h - tmp_h) / 2.0);
    
        memcpy(dst.data + index_h * dst_w * 3, tmp.data, tmp_w * tmp_h * 3);// 缩放后的图片放到 dst 中间，上下留白
        effect_area.x = 0;                  // effect_area表明了缩放后的原图在dst中的区域 x, y, w, h
        effect_area.y = index_h;
        effect_area.width = tmp_w;
        effect_area.height = tmp_h;
    }
    else {
        printf("error\n");
    }
    return 0;
}


// 原图，推理得到的bbox，有效区域effect_roi
void NanoDet::draw_bboxes(const cv::Mat& bgr, const std::vector<BoxInfo>& bboxes, const std::vector<PtsInfo>& points)
{
    // 类别名称
    static const char* class_names[] = {
                                        "B_G", "B_1", "B_2", "B_3", "B_4", "B_5", "B_O", "B_Bs", "B_Bb",
                                        "R_G", "R_1", "R_2", "R_3", "R_4", "R_5", "R_O", "R_Bs", "R_Bb",
                                        "N_G", "N_1", "N_2", "N_3", "N_4", "N_5", "N_O", "N_Bs", "N_Bb",
                                        "P_G", "P_1", "P_2", "P_3", "P_4", "P_5", "P_O", "P_Bs", "P_Bb"
    };

    // 拷贝一份图片
    cv::Mat image = bgr.clone();

    // 遍历所有bbox区域
    for (size_t i = 0; i < bboxes.size(); i++)
    {
        // 获取bbox信息
        const BoxInfo& bbox = bboxes[i];
        const PtsInfo& pts = points[i];
        
        // 根据预测类别label标签，获取该类别对应的一种颜色
        cv::Scalar color = cv::Scalar(color_list[bbox.label][0], color_list[bbox.label][1], color_list[bbox.label][2]);
        //fprintf(stderr, "%d = %.5f at %.2f %.2f %.2f %.2f\n", bbox.label, bbox.score,
        //    bbox.x1, bbox.y1, bbox.x2, bbox.y2);

        // 在原图画出来矩形 bbox坐标是320x320图中的坐标
        // (bbox.x1 - effect_roi.x) * width_ratio 得到原图中的横坐标， 其中bbox.x1 - effect_roi.x 得到的是相对感兴趣区域的横坐标
        // (bbox.y1 - effect_roi.y) * height_ratio 得到原图中的纵坐标  其中bbox.x1 - effect_roi.x 得到的是相对感兴趣区域的纵坐标
        cv::rectangle(image, cv::Rect(cv::Point(bbox.x1, bbox.y1), cv::Point(bbox.x2, bbox.y2)), color);

        cv::line(image, cv::Point(pts.x1, pts.y1), cv::Point(pts.x2, pts.y2), color);
        cv::line(image, cv::Point(pts.x2, pts.y2), cv::Point(pts.x3, pts.y3), color);
        cv::line(image, cv::Point(pts.x3, pts.y3), cv::Point(pts.x4, pts.y4), color);
        cv::line(image, cv::Point(pts.x4, pts.y4), cv::Point(pts.x1, pts.y1), color);
        
        // 文本，标出类别提及得分
        char text[256];
        sprintf(text, "%s %.1f%%", class_names[bbox.label], bbox.score * 100);

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.4, 1, &baseLine);

        // 在bbox的左上角标明文本
        int x = bbox.x1;
        int y = bbox.y1 - label_size.height - baseLine;
        
        if (y < 0)
            y = 0;
        if (x + label_size.width > image.cols)
            x = image.cols - label_size.width;

        // 标出文本框
        cv::rectangle(image, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
            color, -1);
        
        // 在图片上标注文本
        cv::putText(image, text, cv::Point(x, y + label_size.height),
            cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255));
    }
    cv::imshow("image", image);
}

