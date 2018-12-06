import training_functions
import argparse
if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('-m', '--model', type=str, required=False, default='all',
                      help="Which model to train, between classifier and encoder, or only_classifier, or all")
  args = parser.parse_args()

  model_type = args.model
  if model_type == "all":
    training_functions.train_auto_encoder()
    training_functions.train_classifier()
    training_functions.train_only_classifier()
  elif model_type == "classifier":
    training_functions.train_classifier()
  elif model_type == "encoder":
    training_functions.train_auto_encoder()
  elif model_type == "only_classifier":
    training_functions.train_only_classifier()
  else:
    print("Please enter a correct model to train")
