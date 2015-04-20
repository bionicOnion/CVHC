#include "NeuralNet.h"
#include "CharacterExtractor.h"
#include "opencv2/highgui/highgui.hpp"
#include <stdio.h>

int main(int argc, char** argv)
{
  if (argc < 3)
  {
    printf("Usage: %s [classifier] [image]\n", argv[0]);
    return 0;
  }
  
  NeuralNet nn;
  CharacterExtractor ce;

  // Load the neural network from the specified classifier file
  if (!nn.loadNN(argv[1]))
  {
    fprintf(stderr, "There were errors while loading the neural network.\n");
    return 0;
  }

  int i;
  for (i = 2; i < argc; ++i)
  {
    cv::Mat image = cv::imread(argv[2], CV_LOAD_IMAGE_GRAYSCALE);
    if (!image.data)
    {
      printf("The specified image could not be found.\n");
      return 0;
    }
    printf("Prediction: %c\n", nn.classify(image));
  }

  // // Load the specified image
  // cv::Mat srcImage = cv::imread(argv[2], CV_LOAD_IMAGE_COLOR);
  // if (!srcImage.data)
  // {
  //   printf("The specified image could not be found.\n");
  //   return 0;
  // }

  // // Preprocess the image
  // int errCode;
  // cv::Mat destImage, threshImage, croppedImage;
  // std::vector<cv::Rect> boundingBoxes, charBoundingBoxes;
  // if ((errCode = ce.preprocessImage(srcImage, destImage, threshImage, false)) != 0)
  // {
  //   printf("Preprocessing failed with code %i\n", errCode);
  //   return errCode;
  // }
  // if ((errCode = ce.findBoundingBoxes(destImage, boundingBoxes)) != 0)
  // {
  //   printf("Finding bounding boxes failed with code %i\n", errCode);
  //   return errCode;
  // }
  // if ((errCode = ce.findFullCharBoxes(boundingBoxes, charBoundingBoxes)) != 0)
  // {
  //   printf("Correcting bounding boxes failed with code %i\n", errCode);
  //   return errCode;
  // }

  // // Classify every character found within the image
  // std::vector<cv::Rect>::iterator iter;
  // for (iter = charBoundingBoxes.begin(); iter != charBoundingBoxes.end(); ++iter)
  // {
  //   if ((errCode = ce.cropImage(threshImage, croppedImage, *iter)) != 0)
  //   {
  //     printf("Cropping failed with code %i\n", errCode);
  //     return errCode;
  //   }

  //   char label = nn.classify(croppedImage);
  //   printf("Prediction: %c\n", label);
  // }
  return 0;
}