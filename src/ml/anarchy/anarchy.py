import os
import platform
import ml.anarchy.data as traindata

print("importing pandas")
import pandas as pd
print("importing tensorflow")
import tensorflow as tf
print("importing keras")
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.utils import to_categorical
# print("importing sklearn")
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score, precision_score, recall_score
# from sklearn.model_selection import train_test_split


def clear():
    if platform.system().lower() == "windows":
        os.system("cls")
    else:
        os.system("clear")


clear()

def ask():
    clear()
    society = input("What is the name of your society?\n[string]\n >>> ")
    clear()
    political_stability = input("How is the political stability?\n[low, med, high]\n >>> ")
    clear()
    police_strength = input("What is the strength/effectiveness of the police?\n[low, med, high]\n >>> ")
    clear()
    weapon_access = input("Do citizens have easy access to weapons?\n[yes, no]\n >>> ")
    clear()
    herd_mentality = input("What is the herd mentality like?\n[low, med, high]\n >>> ")
    clear()
    ideology = input("What is/are the ideology(ies)?\n[democracy, anarchism, facism, marxism, monarchy, autocracy, jacobinism, communism, socialism, parliamentarian democracy, constitutional monarchy]\n\nEx.\n['socialism', 'marxism']\n['democracy', 'jacobinism']\n >>> ")
    clear()
    try:
        epochs = int(input("How many epochs?\n[integer]\n >>> "))
    except ValueError:
        input(f"epochs must be an int, not type {type(epochs).__name__}\n\epochs will now be 50.\n\nPress any key to continue...")
        epochs = 50

    """
    - Political stability: Low, Medium, High (3 values)
    - Police strength: Low, Medium, High (3 values)
    - Easy weapon access: Yes, No (2 values)
    - Herd Mentality: Low, Medium, High (3 values)
    - Ideology: Democracy, Anarchism, Fascism (3 values)
    """

    prediction = train(
        political_stability=political_stability,
        police_strength=police_strength,
        weapon_access=weapon_access,
        herd_mentality=herd_mentality,
        ideology=ideology,
        epochs=epochs
    )
    clear()

    print(f"""
Political stability: {political_stability}
Police strength....: {police_strength}
Weapon Access......: {weapon_access}
Herd Mentality.....: {herd_mentality}
Ideology...........: {ideology}
Epochs.............: {epochs}

Confidence that "{society}" WILL fall into anarchy: {round(prediction[2]*100, 3)}%
                Will "{society}" fall into anarchy? {prediction[0]}

Dataframe:
{prediction[1]}
""")


def train(political_stability, police_strength, weapon_access, herd_mentality, ideology, epochs):
    data = pd.DataFrame(
        columns=[
            "Political Stability",
            "Police Strength",
            "Weapon Access",
            "Herd Mentality",
            "Ideology",
            "Anarchy",
        ]
    )


    user_input = {
        "Political Stability": political_stability,
        "Police Strength": police_strength,
        "Weapon Access": weapon_access,
        "Herd Mentality": herd_mentality,
        "Ideology": ideology,
        "Anarchy": "?",
    }

    """
    user_input_df = pd.DataFrame(user_input, index=[0])
    data = pd.concat([data, user_input_df], ignore_index=True)

    # Combine the anarchy_societies and societies lists into a single DataFrame
    all_societies = anarchy_societies + societies
    all_societies = [{key: ",".join(val) if isinstance(val, list) else val for key, val in society.items()} for society in all_societies]
    societies_df = pd.DataFrame(all_societies)

    # Convert categorical variables to dummy variables
    X = pd.get_dummies(societies_df.drop("Anarchy", axis=1))

    # Split the data into features (X) and target (y)
    y = societies_df["Anarchy"]

    # Train a logistic regression model on the combined list of societies
    model = LogisticRegression()
    model.fit(X, y)

    # Use the trained model to predict the probability that the user input society will fall into anarchy
    user_input_df = user_input_df.reindex(columns=X.columns, fill_value=0)  # Ensure feature names are consistent
    proba = model.predict_proba(user_input_df)

    # Extract the probability of the 'yes' class
    confidence = proba[0][1]

    # Use the confidence value to determine the predicted class ('yes' or 'no')
    prediction = "yes" if confidence >= 0.5 else "no"

    # Add the confidence and probability estimates to the user input DataFrame
    user_input_df["Confidence"] = confidence
    user_input_df["Probability of No Anarchy"] = proba[0][0]
    user_input_df["Probability of Anarchy"] = proba[0][1]

    return prediction, user_input_df
    """
    user_input_df = pd.DataFrame(user_input, index=[0])

    # Combine the anarchy_societies and societies lists into a single DataFrame
    anarchy_societies = traindata.anarchy_societies
    societies = traindata.societies

    all_societies = anarchy_societies + societies
    all_societies = [{key: ",".join(val) if isinstance(val, list) else val for key, val in society.items()} for society in all_societies]
    societies_df = pd.DataFrame(all_societies)

    # Convert categorical variables to dummy variables
    X = pd.get_dummies(societies_df.drop("Anarchy", axis=1))

    # Ensure that the user input DataFrame has the same columns as the training data
    user_input_df = user_input_df.reindex(columns=X.columns, fill_value=0)

    # Normalize the input data
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    X = (X - mean) / std

    # Apply the same normalization to the user input
    user_input_df = (user_input_df - mean) / std

    # Split the data into features (X) and target (y)
    y = societies_df["Anarchy"]
    y = y.map({"no": 0, "yes": 1})  # Convert string labels to integers
    y = to_categorical(y)  # Convert integer labels to one-hot encoded vectors

    # Define the neural network model
    model = Sequential()
    model.add(Dense(32, input_dim=X.shape[1], activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(2, activation="softmax"))  # Use softmax activation for output layer with 2 classes

    # Compile the model
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    # Train the model on the combined list of societies
    model.fit(X, y, epochs=epochs, batch_size=32, validation_split=0.2)

    # Use the trained model to predict the probability that the user input society will fall into anarchy
    user_input_df = user_input_df.reindex(columns=X.columns, fill_value=0)  # Ensure feature names are consistent
    proba = model.predict(user_input_df)

    # Extract the probability of the 'yes' class
    confidence = proba[0][1]
    print(confidence)

    # Use the confidence value to determine the predicted class ('yes', 'no', or 'maybe)
    if confidence < .5:
        prediction = "no"
    elif confidence >= .6:
        prediction = "yes"
    else:
        prediction = "maybe"

    # Add the confidence and probability estimates to the user input DataFrame
    user_input_df["Confidence"] = confidence

    return prediction, user_input_df, confidence

"""
Political stability: high
Police strength....: high
Weapon Access......: no
Herd Mentality.....: low
Ideology...........: ['democracy']
Epochs.............: 50

Will "Society X" fall into anarchy? no

-----------------------------------------

Political stability: high
Police strength....: med
Weapon Access......: yes
Herd Mentality.....: low
Ideology...........: ['democracy']
Epochs.............: 1000

Will "Society X" fall into anarchy? no
"""
