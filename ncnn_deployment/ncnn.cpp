#include <iostream>
#include "mat.h"
#include "net.h"
#include <fstream>
#include <sstream>
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;

vector<vector<double> > loadImgs(int count=1){
    fstream file;
    string line="",temp;
    double val;
    vector<double> vec;
    vector<vector<double> > imgs;
    file.open("../../mnist-sign-language/mnist_sign_language_test.csv",ios::in);
    getline(file, line);//Get rid of the coloumn names

    while(count>0){
        getline(file, line);
        stringstream s(line);

        while(getline(s, temp, ',')){
            val = stod(temp);
            vec.push_back(val);
        }

        imgs.push_back(vec);
        vec.clear();
        count--;
    }

    return imgs;
}

void getimage(vector<double> v, int img[28][28]){
    for(int i=0;i<28;++i){
        for(int j=0;j<28;++j){
            img[i][j] = v.at(i*28+j+1); //Shift indexing by 1 because the first element is the label
        }
    }
}

void getimage(vector<double> v, unsigned char img[784]){
    for(int i=1;i<785;++i){
            img[i-1] = (unsigned char) v.at(i); //Shift indexing by 1 because the first element is the label
    }
}

//For checking
void print_vectors(vector<vector<double> >v){
    int count=0;
    for(vector<double> i:v){
        cout<<"Entry: "<<count+1<<endl;
        for(double j:i){
            cout<<j<<" ";
        }
        cout<<"\n\n";
        count++;
    }
}

void print_image(int img[28][28]){
    cout<<"Original: "<<endl;
    for(int i=0;i<28;++i){
        for(int j=0;j<28;++j){
            cout<<img[i][j]<<" ";
        }
        cout<<endl;
    }
}

void print_image(unsigned char img[784]){
    cout<<"Original: "<<endl;
    for(int i=0;i<784;++i){
        cout<<(int) img[i]<<" ";
    }
}

void print_mat(cv::Mat mat){
    cout<<"\n\nMAT: "<<endl;
    cout<<mat<<endl;
}

int max_score(vector<float> arr){
    int max = arr.at(0), index=0;
    for(int i=1;i<26;++i){
        if(arr.at(i)>max){
            max = arr.at(i);
            index = i;
        }
    }
    return index;
}

int main(){
    ncnn::Net net;
    net.load_param("../../ncnn_out/resnet18.param");
    net.load_model("../../ncnn_out/resnet18.bin");

    vector<vector<double> > data = loadImgs(100);
    // print_vectors(data); 

    vector<double> datum = data.at(4);
    unsigned char img[784];
    int label;

    label = datum.at(0);
    getimage(datum, img);
    // print_image(img);


    // cv::Mat uint8_img,image = cv::Mat (28,28,CV_8U,&img);
    // image.convertTo(uint8_img, CV_8U);

    // cout<<"Label: "<<label<<endl;
    // cv::imshow("image",uint8_img);
    // cv::waitKey(0);


    //Convert images
    unsigned char* rgbdata = img;// data pointer to RGB image pixels
    int w=28;// image width
    int h=28;// image height



    ncnn::Mat out, in = ncnn::Mat::from_pixels(rgbdata, ncnn::Mat::PIXEL_GRAY, w, h);

    ncnn::Extractor ex = net.create_extractor();
    ex.input("input.1", in);
    ex.extract("36", out);

    ncnn::Mat out_flatterned = out.reshape(out.w * out.h * out.c);
    std::vector<float> scores;
    scores.resize(out_flatterned.w);
    for (int j=0; j<out_flatterned.w; j++)
    {
        scores[j] = out_flatterned[j];
    }
    cout<<"Label: "<<label<<endl;
    cout<<"Predicted: "<<max_score(scores)<<endl;
    
    return 0;
}