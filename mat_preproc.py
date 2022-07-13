import scipy
import numpy as np

# wrapper class for data preprocessing
class preproc:
    
    # class attributes
    source_info = ["SC", "CR", "SI", "M", "FA"]
    response_info = ["RS", "RO", "F", "MN", "SN"]
    
    def __init__(self, file_path):
        
        data = scipy.io.loadmat(file_path)
        # user trail order
        self.tr_order = data['user_tr_order_1'][0]

        # projection scores
        self.proj_score = data['user_prob_1'][0]

        # source and response label
        self.source_label = data['user_source_1'][0]
        self.resp_label = data['user_resp_1'][0]

        # features, group the channels and average over the windows,
        # those are called features, and we use the features for training
        self.behav_feat = data['user_feat_1'][0]
        
        self.data = data
    
    def filter_index(self, pos_source_label: int, pos_resp_label: int,
                           neg_source_label: int, neg_resp_label: int):
        """
        A simplified version of prepare label. Instead of returns all the
        dataset (X, y, group), this only returns a boolean array of the corresponding index
        Since the data is in a nested array (in other words, 2-d array with different dim),
        the returned element should also be a nested array.
        
        The purpose of this is to prepared for multi-subclass merge for a single class preparation
        
        Parameters:
        -----------
        pos_source_label : int
            the positive class's source label.
            for details, please refer to the above encodings
        pos_resp_label : int
            the positive class's response label
        neg_source_label : int
            the negative class's source label
        neg_response_label : int
            the negative class's response label

        Returns:
        --------
        pos_idx : np.ndarray
            the nested boolean array that indicates the location of
            the positive class.
        neg_idx : np.ndarray 
            the nested boolean array that indicates the location of
            the negative class.
        """
        pos_idx, neg_idx = [], []
        
        for source, response, behavior_feature in zip(
            self.source_label, self.resp_label, self.behav_feat
        ):
            # use the logical intersection to subtract out the indices 
            # of the positive and negative class
            pos_index_single_subject = (
                (source.flatten()==pos_source_label) &
                (response.flatten()==pos_resp_label)
                        )
            neg_index_single_subject = (
                (source.flatten()==neg_source_label) & 
                (response.flatten()==neg_resp_label)
            )
            # aggregate back
            pos_idx.append(pos_index_single_subject)
            neg_idx.append(neg_index_single_subject)
        
        return np.array(pos_idx, dtype=object), np.array(neg_idx, dtype=object)

    def filter_index(self, source_label: int, resp_label: int):
        """
        Variant of the above's filter_index. This method get out the indices of a 
        specific class (single class indexer)

        Parameters:
        -----------
        source_label : int
            source label.
            for details, please refer to the above encodings
        resp_label : int
            response label

        Returns:
        --------
        idx : np.ndarray
            the nested boolean array that indicates the location of
            the positive class.
        """
        idx = []
        
        for source, response, behavior_feature in zip(
            self.source_label, self.resp_label, self.behav_feat
        ):
            # use the logical intersection to subtract out the indices 
            # of the positive and negative class
            index_single_subject = (
                (source.flatten()==source_label) &
                (response.flatten()==resp_label)
            )
            # aggregate back
            idx.append(index_single_subject)
        
        return np.array(idx, dtype=object)

    
    def merge_two_class(self, pos1, neg1, pos2, neg2):
        """
        Apply logical OR to two positive class and two negative class
        Purpose is to merge 1 and 2
        
        Parameters:
        -----------
        pos1 : np.ndarray([Object])
            the positive class 1 index array
        neg1 : np.ndarray([Object])
            the negative class 1 index array
        pos2 : np.ndarray([Object])
            the positive class 2 index array
        neg2 : np.ndarray([Object])
            the negative class 2 index array
            
        Returns:
        --------
        pos_idx : np.ndarray
            the merged nested boolean array that indicates the location of
            the positive class.
        neg_idx : np.ndarray 
            the merged nested boolean array that indicates the location of
            the negative class.
        """
        pos_idx, neg_idx = [], []
        for p1, n1, p2, n2 in zip(pos1, neg1, pos2, neg2):
            pos_idx.append((p1 | p2))
            neg_idx.append((n1 | n2))
        return np.array(pos_idx, dtype=object), np.array(neg_idx, dtype=object)
    
    def get_data_by_index(self, pos_idx, neg_idx):
        """
        given positive and negative index array, indexing out the
        given data matrices and flattern them out

        This will excluding the subject with
            10 or less trials on each class.
        
        Parameters:
        -----------
        pos_idx : np.ndarray
            the nested boolean array that indicates the position of the
            positive class
        neg_idx : np.ndarray
            the nested boolean array that indicates the position of the
            negative class
        
        Returns:
        --------
        X : np.ndarray
            the input for the formatted flattern data
        y : np.ndarray
            the ground truth label
        subject : np.ndarray
            th subject number that corresponds to the data_x and data_y
        """
        X, y, subject = np.array([]), np.array([]), np.array([])
        
        for subject_num, zipped in enumerate(zip(pos_idx, neg_idx, self.behav_feat)):
            pos, neg, behavior_feature = zipped
            # the num of pos and neg class is their count of True
            # in the boolean array
            pos_len, neg_len = pos.sum(), neg.sum()

            if pos_len < 10 or neg_len < 10:
                # if this subject has less that 10 trails on 
                # each class of interests
                continue
            
            # append positive class
            try: 
                X = np.vstack([X, behavior_feature[pos, :]])
            except ValueError:
                # catch the first case where the X is empty
                X = behavior_feature[pos, :]
            y = np.append(y, np.repeat(1, pos_len))
            
            # append negative class
            X = np.vstack([X, behavior_feature[neg, :]])
            y = np.append(y, np.repeat(-1, neg_len))

            # record their subject id
            subject = np.append(subject, np.repeat(subject_num, pos_len + neg_len))
        return X, y, subject