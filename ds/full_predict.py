import predict_functions
import argparse
if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('-m', '--model', type=str, required=False, default='classifier',
                      help="Which model to run predictions on, between classifier and encoder, or only_classifier")
  parser.add_argument('-f', '--predict_file', type=str,
                      required=True, help="File to extract prediction data from")
  args = parser.parse_args()

  predict_file_name = args.predict_file
  model_type = args.model

  if model_type == "encoder":
    encodings = predict_functions.test_auto_encoder(predict_file_name)
  elif model_type == "classifier":
    classes = predict_functions.predict_classifier(predict_file_name)
  elif model_type == "only_classifier":
    classes = predict_functions.predict_only_classifier(predict_file_name)
  else:
    print("Please enter either classifier or encoder under -m flag")
