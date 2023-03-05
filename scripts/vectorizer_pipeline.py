from scripts import utils
from sklearn.model_selection import train_test_split
import joblib

VECTORIZER_FOLDER = 'vectorizer_data'
class VectorizerPipeline:
    '''
    This class sets up a vectorizer pipeline that takes an sklearn vectorizer
    and fits and transforms the split data and dumps both the vectorizer and the train
    test splits into appropriate folders. These can then be used for modelling later. 
    '''
    def __init__(
            self, vectorizer_name, vectorizer, 
            X, y
            ):
        # Name of the vectorizer.
        self.vectorizer_name = vectorizer_name
        # The vectorizer object itself.
        self.vectorizer = vectorizer
        self.split_data = self._splitter(X, y)
        self.vectorizer_path = utils.get_datapath(VECTORIZER_FOLDER) / vectorizer_name
        self.transformed_data = {}

    def _dump_vectorizers(self):
        with open(
            self.vectorizer_path / f'{self.vectorizer_name}.pkl', 
            'wb'
        ) as f:
            joblib.dump(self.vectorizer, f)
        
        print(f'Vectorizer dumped at {self.vectorizer_path}/{self.vectorizer_name}.pkl')


    def _dump_test_train_split(self):
        with open(
            self.vectorizer_path / 'data.pkl',
            'wb'
        ) as f:
            joblib.dump(self.transformed_data, f)

        print(f'Transformed train test split dumped at {self.vectorizer_path}/data.pkl as a dictionary.')


    def _splitter(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        return { 
            entry[0] : entry[1] 
            for entry in zip(
                ['X_train', 'X_test', 'y_train', 'y_test'],
                [X_train, X_test, y_train, y_test]
            )
        }
            
        
    def run_vectorizer_pipeline(self):
        self.vectorizer.fit(self.split_data['X_train'])
        X_train_transformed = self.vectorizer.transform(self.split_data['X_train'])
        X_test_transformed = self.vectorizer.transform(self.split_data['X_test'])

        print(
            f'Train shape: {X_train_transformed.shape} \
            \nTest shape: {X_test_transformed.shape}'
        )

        self.transformed_data['X_train'] = X_train_transformed
        self.transformed_data['X_test'] = X_test_transformed
        self.transformed_data['y_train'] = self.split_data['y_train']
        self.transformed_data['y_test'] = self.split_data['y_test']

        self._dump_vectorizers()
        self._dump_test_train_split()