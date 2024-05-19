#include <iostream>
#include "mat.h"
#include "net.h"
#include <fstream>
#include <sstream>
#include <vector>
#include <chrono>
#include <ctime>

// //For debugging
// #include <opencv2/core.hpp>
// #include <opencv2/imgcodecs.hpp>
// #include <opencv2/highgui.hpp>
// #include <opencv2/imgproc/imgproc.hpp>

using namespace std;

ncnn::Net net;
double duration=0;

vector<vector<double> > loadImgs(int count = -1){
    fstream file;
    string line="",temp;
    double val;
    vector<double> vec;
    vector<vector<double> > imgs;
    file.open("../../mnist-sign-language/mnist_sign_language_test.csv",ios::in);
    getline(file, line);//Get rid of the coloumn names

    int decrement=1;
    if(count<0)decrement = 0;
    while( getline(file, line) && abs(count)>0){
        stringstream s(line);

        while(getline(s, temp, ',')){
            val = stod(temp);
            vec.push_back(val);
        }

        imgs.push_back(vec);
        vec.clear();
        count-=decrement;
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

// void print_mat(cv::Mat mat){
//     cout<<"\n\nMAT: "<<endl;
//     cout<<mat<<endl;
// }

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

int infer(vector<double> datum){
    unsigned char img[784];

    //Convert image
    getimage(datum, img);
    ncnn::Mat out, in = ncnn::Mat::from_pixels(img, ncnn::Mat::PIXEL_GRAY, 28, 28);

    //Inferring
    ncnn::Extractor ex = net.create_extractor();
    auto start = std::chrono::system_clock::now();
    ex.input("input.1", in);
    auto end = std::chrono::system_clock::now();
    ex.extract("36", out);

    //Timing processes
    chrono::duration<double> elapsed_seconds = end-start;
    duration+=elapsed_seconds.count();


    //Results
    ncnn::Mat out_flatterned = out.reshape(out.w * out.h * out.c);
    std::vector<float> scores;
    scores.resize(out_flatterned.w);
    for (int j=0; j<out_flatterned.w; j++)
    {
        scores[j] = out_flatterned[j];
    }
    return max_score(scores);
}

int main(){
    system("python3 ../convert.py");
    net.load_param("../ncnn_out/model.param");
    net.load_model("../ncnn_out/model.bin");

    vector<vector<double> > data = loadImgs();
    int count = 0;
    for(int i=0;i<data.size();++i){
        vector<double> datum = data.at(i);
        int prediction = infer(datum),
        label = datum.at(0);
        if(label == prediction) count++;
    }

    cout<<"Accuracy: "<<count/(data.size()*1.0)*100<<"%"<<endl;
    cout<<"Model Latency: "<<duration/(data.size()*1.0)*1000<<" ms"<<endl;

    // // if want to see the images. Please place accordingly
    // cv::Mat uint8_img,image = cv::Mat (28,28,CV_8U,&img);
    // image.convertTo(uint8_img, CV_8U);   
    // cout<<"Label: "<<label<<endl;
    // cv::imshow("image",uint8_img);
    // cv::waitKey(0);
    return 0;
}