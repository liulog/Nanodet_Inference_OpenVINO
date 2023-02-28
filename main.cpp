#include "Detector.h"
#include <string.h>

int main(int argc, char** argv)
{
    std::string img_name="/home/jingyu/pic/0009.png";

    std::cout<<"start init model"<<std::endl;
    
    // 初始化部分
    auto detector = NanoDet("/home/jingyu/nanodet_model/nanodet.xml");        // openvino套件优化后生成的xml

    // 读取图片
    cv::Mat image = cv::imread(img_name);
    
    // 保存结果的数据(处理过的，bbox和pts存放的都是原图上的坐标)
    std::vector<BoxInfo> bbox_dets;
    std::vector<PtsInfo> pts_dets;

    // image_size: 1280x1024
    std::cout<<image.cols<<" "<<image.rows<<std::endl;

    // 利用模型进行detect（处理过的，image输入原图即可）
    detector.detect(image, 0.4, 0.5, bbox_dets, pts_dets);
    
    // 绘制出结果（image输入原图即可）
    detector.draw_bboxes(image, bbox_dets, pts_dets);
    cv::waitKey(0);
}
