#include "NeuralNet.h"

mxArray* NeuralNet::extractElement(mxArray* structure, std::string fieldname)
{
  int numel = mxGetNumberOfElements(structure);
  int i;
  for (i = 0; i < numel; ++i)
  {
    mxArray* extracted = mxGetField(structure, i, fieldname.c_str());
    if (extracted != NULL)
      return extracted;
  }
  return NULL;
}

bool NeuralNet::buildNetwork(mxArray* field_theta, mxArray* field_netconfig)
{
  double* theta = mxGetPr(field_theta);
  mxArray* depth = extractElement(field_netconfig, "layersizes");
  if (depth == NULL)
  {
    printf("The netconfig field is improperly formatted\n");
    return false;
  }
  numLayers = mxGetNumberOfElements(depth);

  double smax[hiddenSize*numClasses];
  std::memcpy(smax, theta, hiddenSize*numClasses);
  softmaxTheta = cv::Mat(numClasses, hiddenSize, CV_64F, smax);

  int prevLayerSize = inputSize, layerSize = 0;
  int i, wLength, bLength;
  int index = hiddenSize*numClasses;
  for (i = 0; i < numLayers; ++i)
  {
    mxArray* field_layerSize = mxGetCell(depth, i);
    layerSize = (int) *(mxGetPr(field_layerSize));

    wLength = layerSize * prevLayerSize;
    bLength = layerSize;

    double w[wLength];
    double b[bLength];
    std::memcpy(w, theta + index, wLength);
    cv::Mat* weights = new cv::Mat(layerSize, prevLayerSize, CV_64F, w);
    stackW.push_back(weights);

    index += wLength;
    std::memcpy(b, theta + index, bLength);
    index += bLength;
    stackB.push_back(new cv::Mat(layerSize, 1, CV_64F, b));

    prevLayerSize = layerSize;
  }
  return true;
}

bool NeuralNet::loadNN(std::string filename)
{
  /*
   * Open the classifier for reading
   */
  MATFile* matfile = matOpen(filename.c_str(), "r");

  if (matfile == NULL)
  {
    printf("The specified classifier file does not exist\n");
    return false;
  }

  /*
   * Extract the classifier struct from the file
   */
  mxArray* classifier = matGetVariable(matfile, "classifier");
  if (classifier == NULL)
  {
    printf("The classifier was not loaded successully\n");
    return false;
  }

  /*
   * Extract the individual elements from the struct
   */
  std::vector<std::string> failed;

  mxArray* field_theta = extractElement(classifier, "theta");
  if (field_theta == NULL) failed.push_back("theta");
  mxArray* field_inputSize = extractElement(classifier, "inputSize");
  if (field_inputSize == NULL) failed.push_back("inputSize");
  mxArray* field_hiddenSize = extractElement(classifier, "hiddenSize");
  if (field_hiddenSize == NULL) failed.push_back("hiddenSize");
  mxArray* field_numClasses = extractElement(classifier, "numClasses");
  if (field_numClasses == NULL) failed.push_back("numClasses");
  mxArray* field_netconfig = extractElement(classifier, "netconfig");
  if (field_netconfig == NULL) failed.push_back("netconfig");

  if (failed.size() > 0)
  {
    printf("The following elements failed to load from the classifier file:\n");
    std::vector<std::string>::iterator iter;
    for (iter = failed.begin(); iter != failed.end(); ++iter)
    {
      printf("\t%s\n", iter->c_str());
    }
    return false;
  }

  /*
   * Convert the extracted elements into the proper format
   */
  inputSize  = (int) *(mxGetPr(field_inputSize));
  hiddenSize = (int) *(mxGetPr(field_hiddenSize));
  numClasses = (int) *(mxGetPr(field_numClasses));
  return buildNetwork(field_theta, field_netconfig);
}

char NeuralNet::classify(cv::Mat image)
{
  if (!preprocess(image))
  {
    printf("The provided image could not be classified\n");
    return '?';
  }
  cv::Mat pred = image;
  cv::Mat w, b;

  int i;
  for (i = 0; i < numLayers; ++i)
  {
    w = *(stackW[i]);
    b = *(stackB[i]);

    pred = sigmoid(w*pred + b);
  }

  pred = softmaxTheta*pred;

  double maxValue = 0, prediction = 0;
  for (i = 0; i < numClasses; ++i)
  {
    if (pred.at<double>(i) > maxValue)
    {
      maxValue = pred.at<double>(i);
      prediction = i;
    }
  }

  return lookup(prediction);
}

bool NeuralNet::preprocess(cv::Mat& image)
{
  image = image.reshape(0,1).t();
  int rows = image.rows;
  image.convertTo(image, CV_64F);
  return (rows == inputSize);
}


cv::Mat NeuralNet::sigmoid(cv::Mat x)
{
  cv::exp(-x, x);
  return 1 / (1 + x);
  return x;
}

char NeuralNet::lookup(int prediction)
{
  if (prediction < 0 || prediction > 62) return lookupTable[0];
  else return lookupTable[prediction];
}