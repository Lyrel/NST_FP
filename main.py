
from nst import NeuralStyleTransfer
import sys
import itertools
from assessment import Assessment
from report import ExperimentationReport
from gui import AcademicResearchGUI
import time
MAX_STEPS = 1000


if __name__ == "__main__":

    def check_scope(parameters):
        """Checks for obvious parameter entry mistakes"""
        # Check if number of steps is out of range
        if parameters["num_steps"] > MAX_STEPS:
            print(f"Error: num_steps ({parameters["num_steps"]}) exceeds maximum allowed steps ({MAX_STEPS})")
        # Check that only permitted optimizers are used
        if parameters["optimizer"][0].lower() != "adam" and parameters["optimizer"][0].lower() != "lbfgs":
            print(f"Error: wrong optimizer. Only 'adam' or 'lbfgs' are accepted. You entered '{parameters["optimizer"][0]}'.")
            sys.exit(1)  # Exit the program with error code 1
        # Check for non-negativity
        if parameters["style_weight"] <= 0:
            print(f"Error: 'style_weight' is negative number.")
            sys.exit(1)  # Exit the program with error code 1
        # Check for non-negativity
        if parameters["content_weight"] <= 0:
            print(f"Error: 'content_weight' is negative number.")
            sys.exit(1)  # Exit the program with error code 1

    def generate_combinations(parameters):
        """
        Composes all possible combinations of inserted parameter options
        :param parameters: dictionary, where values are represented as lists of parameter options
        :return: list of dictionaries, where each represents unique set of NST parameters to run
        """
        keys = parameters.keys()
        values = parameters.values()
        combinations = []
        # makes all possible combinations
        for i, v in enumerate(itertools.product(*values)):
            param_set = dict(zip(keys, v))
            check_scope(param_set)
            # each parameter combination assigned own ID number
            param_set["id"] = i +1
            combinations.append(param_set)
        return combinations

    def run_gui_input():
        """Runs GUI input of NST parameters"""
        app = AcademicResearchGUI()
        # starts the gui main window
        app.gui()
        # NST settings from gui
        combination = app.nst_set_parameters()
        print(combination)

        nst = NeuralStyleTransfer(combination)
        loss = nst.run_style_transfer()
        print(loss)
        ass = Assessment()
        metrics = ass.evaluate_experiment_results(combination, loss)
        report = ExperimentationReport(combination, metrics)
        report.create_report()
    def confirm_manual_run():
        """Asks user for confirmation before running manual input"""
        print("\n=== WARNING: About to run Neural Style Transfer with manual settings ===")
        print("\n=== WARNING: Make sure you updated inputs in run manual input ===")
        print("\nDo you want to proceed? (yes/no): ")

        while True:
            response = input().strip().lower()
            if response in ["yes", "y"]:
                return True
            elif response in ["no", "n"]:
                return False
            else:
                print("Please enter 'yes' or 'no': ")

    def run_manual_input(parameter_grid):

        # generates all possible parameter combinations
        param_combinations = generate_combinations(parameter_grid)

        for combination in param_combinations:
            # runtime timer starts
            start_time = time.time()
            # passes parameters to NST
            nst = NeuralStyleTransfer(combination)
            loss = nst.run_style_transfer()
            # initiates Assessment instance with the model copy from NST
            ass = Assessment()
            # returns dictionary with evaluation metrics
            metrics = ass.evaluate_experiment_results(combination, loss)
            # initiates  PDF report and passes input parameters and assessment metrics
            report = ExperimentationReport(combination, metrics)
            report.create_report()
            # runtime timer ends
            end_time = time.time()
            # runtime calculation
            elapsed_time = end_time - start_time
            print(f"Experiment {combination["id"]} - "
                  f"Steps: {combination["num_steps"]} "
                  f"completed in: {elapsed_time:.2f} seconds")

    # Links to several content and style images
    content_image1 = "./images/content_images/dancing.jpg"
    content_image2 = "./images/content_images/city.jpg"
    style_image1 = "./images/style_images/warhol1.png"
    style_image2 = "./images/style_images/ryabchenko1.png"
    style_image3 = "./images/style_images/barnet1.png"
    style_image4 = "./images/style_images/katz1.png"
    style_image5 = "./images/style_images/keeffe1.png"
    style_image6 = "./images/style_images/monet1.jpg"

    """
    Fill parameter_prid inputs with parameters to run NST.
    
    Parameter options:
    -----------
    'content_image' : [content_image1] 
        A single content image selection, where content_image1 is a path to a jpg or png image file.
        
    'content_image' : [content_image1, content_image2]
        A multi content image selection, where content_image1, content_image2 are paths to a jpg or png image file.
          
    'style_image' : [style_image1] 
        A single style image selection, where style_image1 is a path to a jpg or png image file.
        
    'style_image' : [style_image1, style_image2]
        A multi style image selection, where style_image1, style_image2 are paths to a jpg or png image file. 
    
    'content_layer': ['conv_4']
        A single content layer input
    
    'content_layer': ['conv_4', 'conv_6']
        A multi content layer input
            
    'style_layers': [['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']]
        A single selection of layers for stylization
    
    'style_layers': [['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5'], ['conv_2', 'conv_4', 'conv_6', 'conv_8', 'conv_10']]
        A multiple selections of layers for stylization. Each set of style layers mut represent a list.
   
    'style_weight': [1000000]
        A single option of style weight
    
    'style_weight': [1000000, 1000, 10000]
        A multipleM optionS of style weight
   
    'content_weight': [1]
        A single option of content weight
    
    'content_weight': [1, 5, 10]
        A multiple option of content weight

    'optimizer': ['lbfgs'] or 'optimizer': ['adam', 0.02]
        A single optimizer selection. 0.02 is a default learning rate
    
    'optimizer': [['lbfgs'], ['adam', 0.02]] 
        A multiple optimizer selection. 0.02 is a default learning rate
        
    'num_steps': [200] - single input
    'num_steps': [200, 300, 600] - multiple inputs
    """

    parameter_grid = {
        'content_image': [content_image1, content_image2],
        'style_image': [style_image1, style_image3],
        'content_layer': ["conv_4"],
        'style_layers': [["conv_1", "conv_2", "conv_3", "conv_4", "conv_5"]],
        'style_weight': [100000],
        'content_weight': [1],
        'optimizer': [["lbfgs"]],
        'num_steps': [200]
    }

    # "MANUAL" or any other value will start manual input, "GUI" will GUI input
    input_type = "GUI"

    if input_type == "GUI":
        print("Running GUI")
        run_gui_input()
    else:
        print("Manual input selected")
        if confirm_manual_run():
            print("Starting manual input execution...")
            run_manual_input(parameter_grid)
        else:
            print("Manual input execution cancelled.")
            sys.exit(0)


