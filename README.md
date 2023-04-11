# Incline-Speed-LSTM

This study aims to assess the accuracy and practicality of incorporating machine learning algorithms with instrumented insoles for studying Achilles tendon loading outside laboratory environments by developing three LSTMs that predict treadmill incline and walking speed using ground reaction forces (GRFs) data. 


## Description

Instrumented insoles with 3 force sensors were used to collected data from fifteen participants that were asked to perform treadmill exercises at varying treadmill incline (0-25%) and walking speed (0.8-1.6 m/s) conditions, with 5% and 0.4 m/s increment, respectively. The ground reaction forces (GRFs) were recorded at the forefoot, midfoot, and heel positions during 30-second walking exercise phases. In order to ensure consistency across participants, each subject performed the same set of conditions. GRFs were normalized by bodyweight (kg) to standardize the data and ensure accurate comparison between participants. A pre-existing algorithm was used to calculate the loading on the Achilles tendon for each subject.The individual steps were extracted from the time series sequences using a peak detection algorithm. The extracted time series sequences for the steps were then fed into a 3, 6, 18 class Long Short Term Memory Network (LSTM). The accuracy of the developed models was evaluated based on their speed and incline prediction performance (80-20 train validation split). This code intends to display the potential of using machine learning and instrumented insoles for predicting Achilles tendon loading during exercise.
 

## Getting Started

### Dependencies

Python 3

Pandas

Mat4py

### Installing

To run this code, you must have Python installed. You can install Pandas and Mat4py using pip:
-	Pip install pandas
-	Pip install mat4py

Note
This code has been written with a specific data file in mind (data_for_Brigid_7_6_2022.mat), and may not work with other data files without modification.

### Executing program
Data
The preprocessed data is stored in a MATLAB file, which is loaded into a Python dictionary using the mat4py library. 

<img width="157" alt="image" src="https://user-images.githubusercontent.com/121143118/231233487-665d3759-85b7-45f2-9795-df80626dd528.png">
The data consists of time series measurements of three features (Forefoot, Midfoot, and Heel) collected from multiple subjects walking on a treadmill at different inclines and speeds. The goal is to classify each time series based on the incline and speed combination.

<img width="468" alt="image" src="https://user-images.githubusercontent.com/121143118/231233605-85db0041-e76b-4a52-bd76-4acfd03d78d4.png">
Preprocessing
The code extracts time series data from the dictionary and constructs time series of 85 time steps and 3 features. The time series data is then appended to a 2D list along with its corresponding label, which is an integer between 0 and 17, representing the incline and speed combination. The label_list and timeseries_list are constructed and converted to numpy arrays. More specifically, the first for loop loops through each subject. Within this loop, a label is set to 0 to keep track of the speed. The next two for loops loop through each incline and each speed, respectively (the try and except statements are used to handle cases where there is no data available for the current incline and speed for the current subject. If this is the case, the continue statement is executed, and the loop moves on to the next incline and speed).

<img width="308" alt="image" src="https://user-images.githubusercontent.com/121143118/231233658-ec9574d2-d5e2-4dd8-949d-e29689e56161.png">
Note: this part of the code varies depending on how many classes we aim to predict.
Then the code loops through each key-value pair in the subject data. For each key-value pair, the Forefoot, Midfoot, and Heel data are flattened into lists using the list(chain.from_iterable()) method. The first 85 values of each list are taken, and the three lists are combined into one list of length 255. If the length of the combined list is not equal to 255 (i.e., there is missing data), the continue statement is executed, and the loop moves on to the next key-value pair. If there is no missing data, the combined list is appended to the timeseries_list list along with the corresponding label. The label is incremented by 1 for each speed. This process is repeated for each subject, incline, and speed.

<img width="375" alt="image" src="https://user-images.githubusercontent.com/121143118/231233756-c1060b2e-2402-475b-a401-5f5db13af731.png">
The timeseries_list and label_list are then converted to numpy arrays using np.array() and assigned to timeseries_matrix and label_array, respectively. The shape of timeseries_matrix is then printed to confirm that the resulting 2D numpy array has the expected shape.

<img width="245" alt="image" src="https://user-images.githubusercontent.com/121143118/231233902-228ff1a3-3875-4c97-907c-8ef6e31dee7b.png">
Finally, the two-dimensional timeseries_matrix array is transformed into a three-dimensional tensor with 85 time steps and three features per step. The reshape() function is applied to timeseries_matrix with three arguments: -1, 85, and 3. The first argument (-1) instructs NumPy to calculate the size of the first dimension based on the original array and the other two dimensions. The second argument (85) specifies the number of time steps, while the third argument (3) denotes the number of features per step. This reshaping is a crucial step in processing and analyzing the data with machine learning models.

<img width="468" alt="image" src="https://user-images.githubusercontent.com/121143118/231234173-985067bc-0aef-4404-867d-e43dbbc19254.png">
RNN Modeling
The RNN is defined using the Keras API from TensorFlow. The model consists of an LSTM layer with 64 units, followed by two fully connected layers with 32 and 18 units, respectively. The LSTM layer takes in the reshaped time series data as input. The output of the final dense layer is a probability distribution over the 18 possible labels. The sparse_categorical_crossentropy loss function is used, along with the Adam optimizer. The sparse_categorical_accuracy metric is used to evaluate the performance of the model during training.

<img width="351" alt="image" src="https://user-images.githubusercontent.com/121143118/231234252-ca440ab2-0da6-48b1-a70c-880add592f34.png">
Training and Evaluation
The data is split into training and testing sets using the train_test_split function from the sklearn.model_selection library. 

<img width="442" alt="image" src="https://user-images.githubusercontent.com/121143118/231234306-ab332210-589c-4f0e-8ea2-04d496fbc3e1.png">
The model is trained using the fit method, with the training and testing sets as inputs. The training is run for 200 epochs, with a batch size of 32. The validation data is passed as a tuple to the validation_data argument of the fit method.

<img width="468" alt="image" src="https://user-images.githubusercontent.com/121143118/231234376-f548b4d6-89ab-4403-a1c5-ee1e965167f7.png">
After training, the performance of the model is visualized using matplotlib. Two plots are created, one for training accuracy and one for validation accuracy. These plots show the accuracy of the model on the training and validation sets as a function of the number of epochs. The accuracy of the model can be interpreted as the fraction of correctly classified time series in the dataset.

<img width="179" alt="image" src="https://user-images.githubusercontent.com/121143118/231234560-efc788fa-c818-4145-a3aa-ba9aec04584a.png">


## Help

If you encounter any issues while using this project, try the following:

- Check that all dependencies are installed correctly
- Make sure that you are using the correct command to run the project
- Consult the project's documentation or the developer community for help

## Authors

Danushka Bandara

dbandara@fairfield.edu

Laia Vancells Lopez 

laia.vancellslopez@student.fairfield.edu

## Version History
* 0.1
    * Initial Release

## License

This project is licensed under the laiavancells License - see the LICENSE.md file for details

## Acknowledgments

Inspiration, code snippets, etc.
I would like to thank Dr. Drazan for their valuable feedback and contributions to this project. His insights and expertise were instrumental in improving the quality of our work.

