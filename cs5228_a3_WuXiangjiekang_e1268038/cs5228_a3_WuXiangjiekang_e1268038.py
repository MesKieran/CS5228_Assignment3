import numpy as np
from sklearn.tree import DecisionTreeClassifier


class DecisionStumpClassifier:
    
    def __init__(self):
        # Keeps the index of best feature and the value of the best threshold
        self.feature_idx, self.threshold = None, None
        # Keeps the indices of the data samples in the left and right child node
        self.y_left, self.y_right = None, None
        
        
    def calc_thresholds(self, x):
        """
        Calculates the set of all valid thresholds given a list of numerical values.
        The set of all valid thresholds is a set of minimum size that contains
        the values that would split the input list of values into two sublist:
        (1) all values less or equal the the threshold
        (2) all values larger than the threshold

        Inputs:
        - x: A numpy array of shape (N,) containing N numerical values, 
             Example: x = [4, 1, 2, 1, 1, 3]
             
        Returns:
        - Set of numerical values representing the thresholds 
          Example for input above: set([1.5, 2.5, 3.5])
        """              
        
        ## Get unique values to handle duplicates; return values will already be sorted
        values_sorted = np.unique(x)
        
        ## Return the "middle values" as thresholds
        return (values_sorted[:-1] + values_sorted[1:]) / 2.0
        
        
        
    def calc_gini_score_node(self, y):
        """
        Calculates Gini Score of a node in the Decision Tree

        Inputs:
        - y: A numpy array of shape (N,) containing N numerical values representing class labels, 
             Example: x = [0, 1, 1, 0, 0, 0, 2]
             
        Returns:
        - Gini Score of node as numeriv value
        """            
        
        gini = None
        
        ################################################################################
        ### Your code starts here ######################################################
        # Calculate the total number of data points in the node
        total_samples = len(y)
    
        # Get the unique class labels and their counts
        unique_classes, class_counts = np.unique(y, return_counts=True)
    
        # Calculate the Gini impurity for the node
        gini = 1.0
        for class_count in class_counts:
            class_prob = class_count / total_samples
            gini -= class_prob ** 2
        
        ### Your code ends here ########################################################
        ################################################################################        
        
        return gini
    
        
    def calc_gini_score_split(self, y_left, y_right):
        """
        Calculates Gini Score of a split; since we only consider binary splits, 
        this is the weighted average of the Gini Score for both child nodes.

        Inputs:
        - y_left:  A numpy array of shape (N,) containing N numerical values representing class labels, 
                   Example: x = [0, 1, 1, 0, 0, 0, 2]
        - y_right: A numpy array of shape (N,) containing N numerical values representing class labels, 
                   Example: x = [1, 2, 2, 2, 0, 2]
             
        Returns:
        - Gini Score of split as numeric value
        """   
        
        
        split_score = None
        
        ################################################################################
        ### Your code starts here ######################################################
        # Calculate the Gini score for the left child node
        gini_left = self.calc_gini_score_node(y_left)
    
        # Calculate the Gini score for the right child node
        gini_right = self.calc_gini_score_node(y_right)
    
        # Calculate the weighted average of Gini scores for the split
        total_samples = len(y_left) + len(y_right)
        split_score = (len(y_left) / total_samples) * gini_left + (len(y_right) / total_samples) * gini_right
    
        ### Your code ends here ########################################################
        ################################################################################        
        
        return split_score
        
        
    def fit(self, X, y):
        """
        Trains the Decision Stump, i.e., finds the best split w.r.t. all features
        and all possible thresholds

        Inputs:
        - X: A numpy array of shape (N, D) containing N data samples presented by D features,
        - y: A numpy array of shape (N,) containing N numerical values representing class labels
             
        Returns:
        - self
        """           
        
        # Initilize the score with infinity
        score = np.inf
        # Initialize best feature index and threshold
        best_feature_idx = None
        best_threshold = None
        ## Loop through all features (columns of X) to find the best split
        for feature_idx in range(X.shape[1]):
            
            # Get all values for current feature
            values = X[:, feature_idx]
            
            # Loop over all possible threshold; we are keeping it very simple here
            # all possible thresholds (see method above)
            for threshold in self.calc_thresholds(values):
                
                ################################################################################
                ### Your code starts here ######################################################                
                # Split the data based on the current feature and threshold
                y_left = y[values <= threshold]
                y_right = y[values > threshold]
                
                # Calculate the Gini score for the split
                split_score = self.calc_gini_score_split(y_left, y_right)
                
                # Check if this split is better than the best split found so far
                if split_score < score:
                    score = split_score
                    best_feature_idx = feature_idx
                    best_threshold = threshold

        # Set the best feature index and threshold for the Decision Stump
        self.feature_idx = best_feature_idx
        self.threshold = best_threshold
                ### Your code ends here ########################################################
                ################################################################################                      
        ## Return DecisionStumpClassifier object
        return self                 
                
                    
    def predict(self, X):
        """
        Uses Decision Stump to predict the class labels for a set of data points

        Inputs:
        - X: A numpy array of shape (N, D) containing N data samples presented by D features,
             
        Returns:
        - y_pred: A numpy array of shape (N,) containing N integer values representing the predicted class labels
        """
                
        y_pred = np.zeros((X.shape[0], ))
        
        ################################################################################
        ### Your code starts here ######################################################
        # Get the feature values for the best feature index
        values = X[:, self.feature_idx]
    
        # Make predictions based on the threshold
        y_pred[values <= self.threshold] = 0  # Class label 0
        y_pred[values > self.threshold] = 2   # Class label 2
        ### Your code ends here ########################################################
        ################################################################################            
                    
        return y_pred    
    
    
    
    
    
class AdaBoostTreeClassifier:
    
    def __init__(self, n_estimators=50):
        self.estimators, self.alphas = [], []
        self.n_estimators = n_estimators
        self.classes = None
        
    
    def fit(self, X, y):
        """
        Trains the AdaBoost classifier using Decision Trees as Weak Learners.

        Inputs:
        - X: A numpy array of shape (N, D) containing N data samples presented by D features,
        - y: A numpy array of shape (N,) containing N numerical values representing class labels
             
        Returns:
        - self
        """        
        
        N = X.shape[0]
        # Initialize the first sample as the input
        D, d = X, y
        # Initialize the sample weights uniformly
        w = np.full((N,), 1/N)
        # Create the list of class labels from all unique values in y
        self.classes = np.unique(y)
        
        for m in range(self.n_estimators):

            # Step 1: Train Weak Learner on current datset sample
            estimator = DecisionStumpClassifier().fit(D, d)
            
            # If you don't trust your implementation of DecisionStumpClassifier or just for testing,
            # you can use the line below to work with the implementation of sklearn instead
            #estimator = DecisionTreeClassifier(max_depth=1).fit(D, d)
            
            # Add current stump to sequences of all Weak Learners
            self.estimators.append(estimator)
            
            ################################################################################
            ### Your code starts here ######################################################
            
            # We give some guides but feel free to ignore if you have a better solution
            # misclassified, e, a, w below are all assumed to be numpy arrays
            
            # Step 2: Identify all samples in X that get misclassified with the current estimator
            misclassified = np.where(estimator.predict(D) != d)[0]
            # Step 3: Calculate the total error for current estimator
            e = np.sum(w[misclassified]) 
            # Step 4: Calculate amount of say for current estimator and keep track of it
            # (we need the amounts of say later for the predictions)
            e = np.sum(w[misclassified])
            a = 0.5 * np.log((1 - e) / max(e, 1e-10))
            self.alphas.append(a)
            # Step 5: Update the sample weights w based on amount of say a
            w[misclassified] *= np.exp(a)
            # Step 6: Normalize the weights to make them sum up to 1
            w /= np.sum(w)
            # Step 7: Sample next D and d
            indices = np.random.choice(N, N, p=w)
            D, d = X[indices], y[indices]
            ### Your code ends here ########################################################
            ################################################################################            
            
        # Convert the amounts-of-say to a numpy array for convenience
        # We need this later for making our predictions
        self.alphas = np.array(self.alphas)
        
        ## Return AdaBoostTreeClassifier object
        return self         
        

            
    def predict(self, X):
        """
        Predicts the class label for an array of data points

        Inputs:
        - X: A numpy array of shape (N, D) containing N data samples presented by D features,
             
        Returns:
        - y_pred: A numpy array of shape (N,) containing N integer values representing the predicted class labels
        """        
        
        # Predict the class labels for each sample individually
        return np.array([ self.predict_sample(x) for x in X ])
        
        
    def predict_sample(self, x):
        """
        Predicts the class label for a single data point

        Inputs:
        - x: A numpy array of shape (D, ) containing D features,
             
        Returns:
        - y_pred: integer value representing the predicted class label
        """        
        
        y = None
        
        # The predict method of our classifier expects a matrix,
        # so we need to convert our sample to a ()
        x = np.array([x])
        
        # Create a vector for all data points and all n_estimator predictions
        y_estimators = np.full(self.n_estimators, -1, dtype=np.int16)
        
        # Stores the score for each class label, e.g.,
        # class_scores[0] = class score for class 0
        # class_scores[1] = class score for class 1
        # class_scores[2] = class score for class 2
        class_scores = np.zeros(len(self.classes))
        
        y_pred = None
        
        ################################################################################
        ### Your code starts here ######################################################        
        for m in range(self.n_estimators):
            y_estimators[m] = self.estimators[m].predict(x)
            class_scores[y_estimators[m]] += self.alphas[m]

        # Choose the class label with the highest score as the prediction
        y_pred = self.classes[np.argmax(class_scores)]
        ### Your code ends here ########################################################
        ################################################################################
        
        return y_pred    