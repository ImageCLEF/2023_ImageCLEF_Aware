import json
import numpy as np
from scipy.stats.stats import pearsonr

SUBMISSION_FILE_PATH = "my prediction file.json"


# IMAGECLEF 2022 AWARE
class AIcrowdEvaluator:

  REQUIRED_SITUATION_CODES = ("acc", "it", "bank", "wait")


  def __init__(self, ground_truth_path, **kwargs):
    """
    This is the AIcrowd evaluator class which will be used for the evaluation.
    Please note that the class name should be `AIcrowdEvaluator`
    `ground_truth` : Holds the path for the ground truth which is used to score the submissions.
    """
    self.ground_truth_path = ground_truth_path
    self.gt = self.load_gt()
    

  def _evaluate(self, client_payload, _context={}):
    """
    This is the only method that will be called by the framework
    returns a _result_object that can contain up to 2 different scores
    `client_payload["submission_file_path"]` will hold the path of the submission file
    """
    print("evaluate...")
    # Load submission file path
    submission_file_path = client_payload["submission_file_path"]
    # Load preditctions and validate format
    predictions = self.load_predictions(submission_file_path)
    score_pearson_correlation_coefficient = self.compute_primary_score(predictions)
    score_secondary = self.compute_secondary_score(predictions)

    _result_object = {
        "score": score_pearson_correlation_coefficient,
        "score_secondary": score_secondary
    }

    return _result_object


  def load_gt(self):
    """
    Load and return groundtruth data
    """
    print("loading ground truth...")

    with open(self.ground_truth_path) as gt_file:
      gt= json.load(gt_file)

    return gt


  def load_predictions(self, submission_file_path):
    """
    Load and return a predictions object (dictionary) that contains the submitted data that will be used in the _evaluate method
    Validation of the runfile format has to be handled here. simply throw an Exception if there is a validation error.
    """
    print("load predictions...")
    # Format:
    # {
    #   "b04bqghczktsjki8": {
    #       "acc": 22.414681099130608,
    #       "it": 29.959636391760895,
    #       "bank": -0.982905009420304,
    #       "wait": 25.87581294786184
    #   },
    #   ...}
    # 
    # user_profile:
    #  "acc": score   (situation code: accommodation search
    #  "it": score    (situation code: job search in IT
    #  "bank": score  (situation code: bank loan search
    #  "wait": score  (situation code: job search as a waiter
    try:
      with open(submission_file_path) as submission_file:
        user_data_items = json.load(submission_file)
    except Exception as e:
      raise Exception("Error loading submission file. Please make sure your file is formatted as valid JSON. " + 
        "Error: "+str(e))

    allowed_user_profiles = self.gt.keys()
    predictions = {}

    # NBR OF USERPROFILES DOES NOT MATCH WITH NBR USERPROFILES IN GT => ERROR
    if len(user_data_items.keys()) != len(allowed_user_profiles):
      raise Exception("Number of user profiles in submission file not equal to number of user_profiles " +
        "in gt set.",len(user_data_items.keys()))

    record_count = 0
    for user_profile in user_data_items:
      record_count += 1

      # USERPROFILE DOES NOT EXIST IN GT => ERROR
      if user_profile not in allowed_user_profiles :
        self.raise_exception("User profile '{}' does not exist in gt set.", record_count, user_profile)

      # USERPROFILE MORE THAN ONCE IN SUBMISSION => ERROR
      # THIS CHECK IS NOT NEEDED AS the value of same key that occurs later will be override the previous value
      # when JSON is parsed
      # if user_profile in predictions:
      #   self.raise_exception("User profile '{}' contained more than once in submission file", record_count)

      scores = user_data_items[user_profile]

      # NUMBER OF ATTRIBUTES NOT 4 => ERROR
      if len(scores) != 4:
        self.raise_exception("User profile '{}' must contain exactly 4 key/value pairs where "+
          "key=situation_code and value=score (Required situation codes: {}).",
          record_count, user_profile, type(self).REQUIRED_SITUATION_CODES)
      
      # ATTRIBUTE DOES NOT EXIST => ERROR
      for score_name in scores:
        if score_name not in type(self).REQUIRED_SITUATION_CODES:
          self.raise_exception("Situation code '{}' for user profile '{}' does not exist "+
            "(Possible situation codes: {}).",
            record_count, score_name, user_profile, type(self).REQUIRED_SITUATION_CODES)

      score_validation_args = {}
      score_validation_args["user_profile"] = user_profile
      score_validation_args["record_count"] = record_count

      # ACCOMODATION SCORE NOT A NUMBER => ERROR
      # ACCOOMODATION SCORE NOT BETWEEN MIN AND MAX => ERROR
      score_validation_args["situation_code"] = "acc"
      score_validation_args["score"] = scores["acc"]

      self.validate_score(score_validation_args)

      # IT SCORE NOT A NUMBER => ERROR
      # IT SCORE NOT BETWEEN MIN AND MAX => ERROR
      score_validation_args["situation_code"] = "it"
      score_validation_args["score"] = scores["it"]

      self.validate_score(score_validation_args)

      # BANK SCORE NOT A NUMBER => ERROR
      # BANK SCORE NOT BETWEEN MIN AND MAX => ERROR
      score_validation_args["situation_code"] = "bank"
      score_validation_args["score"] = scores["bank"]

      self.validate_score(score_validation_args)

      # WAIT SCORE NOT A NUMBER => ERROR
      # WAIT SCORE NOT BETWEEN MIN AND MAX => ERROR
      score_validation_args["situation_code"] = "wait"
      score_validation_args["score"] = scores["wait"]

      self.validate_score(score_validation_args)

      predictions[user_profile] = {
        "acc": float(scores["acc"]),
        "it": float(scores["it"]),
        "bank": float(scores["bank"]),
        "wait": float(scores["wait"])
      }
    
    return predictions

  def raise_exception(self, message, record_count, *args):
    raise Exception(message.format(*args)+" Error occured at record nbr {}.".format(record_count))

  def validate_score(self, score_validation_args):
    score = score_validation_args["score"]
    score_type = score_validation_args["situation_code"]
    user_profile = score_validation_args["user_profile"]
    record_count = score_validation_args["record_count"]

    # SCORE NOT INT OR FLOAT => ERROR
    if type(score).__name__ != "int" and type(score).__name__ != "float":
      self.raise_exception("Score for situtation code '{}' for user profile '{}' must be " +
        "a number. In case it is a number make sure that you remove the quotes. ",
        record_count, score_type, user_profile)


  def compute_primary_score(self, predictions):
    """
    Compute and return the primary score
    `predictions` : valid predictions in correct format
    NO VALIDATION OF THE RUNFILE SHOULD BE IMPLEMENTED HERE
    Validation should be handled in the load_predictions method
    """
    print("compute primary score...")

    sum_correlation = 0
    situations = {"acc", "it", "bank", "wait"}
    print("compute primary score...")    
    for sit in situations:		
      gt_list = []
      predictions_list = []	
      for user in self.gt:
        #print("   ",gt[user][sit],predictions[user][sit])
        gt_list.append(self.gt[user][sit])
        predictions_list.append(predictions[user][sit])
      sit_correlation = pearsonr(gt_list,predictions_list)
      sum_correlation = sum_correlation+float(sit_correlation[0])
      #print(sit,sit_correlation[0],sum_correlation)
    mean_correlation = sum_correlation/len(situations)

    return mean_correlation


  def compute_secondary_score(self, predictions):
    """
    Compute and return the secondary score
    Ignore or remove this method if you do not have a secondary score to provide
    `predictions` : valid predictions in correct format
    NO VALIDATION OF THE RUNFILE SHOULD BE IMPLEMENTED HERE
    Validation should be handled in the load_predictions method
    """
    print("compute secondary score...")

    return 0.0



if __name__ == "__main__":
    ground_truth_path = "gt_val.json"
    _client_payload = {}
    _client_payload["submission_file_path"] = SUBMISSION_FILE_PATH

    # Instantiate a dummy context
    _context = {}

    # Instantiate an evaluator
    aicrowd_evaluator = AIcrowdEvaluator(ground_truth_path)
    
    # Evaluate
    result = aicrowd_evaluator._evaluate(_client_payload, _context)
    print("")
    print("==========================")
    print("Your data was successfully validated ✓")
    print("result")
    print(result)
    print("==========================")
    
