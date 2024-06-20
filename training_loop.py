
import torch
import torch.nn as nn
import torch.optim as optim
from crnn import CNN_BiGRU_Classifier
import math
from tqdm import tqdm
import numpy as np
from training_data import load_training_data
from sklearn.model_selection import train_test_split
from greedy_decoder import GreedyCTCDecoder
from utils import get_actual_transcript
import torchaudio

device = torch.device("cuda:0")

labels_int = np.arange(11).tolist()
labels = [f"{i}" for i in labels_int] # Tokens to be fed into greedy decoder
greedy_decoder = GreedyCTCDecoder(labels=labels)

# Model Parameters
input_size = 1  # Number of input channels
hidden_size = 128
num_layers = 3
output_size = 11  # Number of output classes
dropout_rate = 0.2

n_sequences = 100
len_sequence = 9000

# Model Definition
model = CNN_BiGRU_Classifier(input_size, hidden_size, num_layers, output_size, dropout_rate)
optimizer = optim.Adam(model.parameters(), lr=0.001)
ctc_loss = nn.CTCLoss()

torch.set_default_device(device)
X, y = load_training_data()

# Creating Train, Test, Validation sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1) 


n_classes = 10 
step_sequence = 150
window_overlap = 50
length_per_sample = 200

epochs = 20

# Add over epochs
for epoch in range(epochs):

    print(f"Epoch {epoch}")

    model.train()
    for i in tqdm(range(len(X_train))):

        training_sequence, target_sequence = X_train[i], torch.tensor(y_train[i]).to(device)

        sequence_length = len(training_sequence)
        
        n_samples = math.ceil(sequence_length/step_sequence) # Since we send the last one even if it is small as can be

        seq_model_output = torch.zeros(n_samples, n_classes+1) # To include the blank token

        ptr = 0
        counter = 0
        while ptr <= sequence_length:
            
            if ptr + length_per_sample > sequence_length:
                sequence_chop = training_sequence[ptr:-1] # For when the window has crossed the end
                pad = np.zeros(length_per_sample - (sequence_length-ptr))
                sequence_chop = np.concatenate((sequence_chop, pad))
            else:
                sequence_chop = training_sequence[ptr:ptr+length_per_sample]
            
            ptr += step_sequence
            
            model_input = torch.tensor(sequence_chop, dtype=torch.float32).view(1, 1, len(sequence_chop)).to(device)
            model.to(device)

            # Zero out the gradients
            optimizer.zero_grad()
            
            model_output_timestep = model(model_input) # Getting model output

            sample_counter = int(ptr/step_sequence) - 1
            seq_model_output[sample_counter,:] = model_output_timestep[0] # Aggregating model output over all the samples to create a sequence output to be fed to the loss function
            
            counter += 1

        input_lengths = torch.tensor(n_samples)
        target_lengths = torch.tensor(len(target_sequence))

        loss = ctc_loss(seq_model_output, target_sequence, input_lengths, target_lengths)
        
        loss.backward()

        # Update the weights
        optimizer.step()

        if i % 20 == 0:
            print(f"Epoch {epoch} Batch {i}")
            print(f"Loss {loss.item()}")
            greedy_result = greedy_decoder(seq_model_output)
            greedy_transcript = " ".join(greedy_result)
            #greedy_wer = torchaudio.functional.edit_distance(actual_transcript, greedy_result) / len(actual_transcript)

            print(f"Transcript: {greedy_transcript}")
            print(f"Actual Transcript: {get_actual_transcript(target_sequence)}")
            #print("Motif Error Rate: {motif_err}")
            

    
    # Validation Loop every epoch
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for i in range(len(X_val)):

            training_sequence, target_sequence = X_val[i], torch.tensor(y_val[i]).to(device)
            sequence_length = len(training_sequence)
        
            n_samples = math.ceil(sequence_length/step_sequence) # Since we send the last one even if it is small as can be

            seq_model_output = torch.zeros(n_samples, n_classes+1) # To include the blank token

            ptr = 0
            counter = 0
            while ptr <= sequence_length:
                
                if ptr + length_per_sample > sequence_length:
                    sequence_chop = training_sequence[ptr:-1] # For when the window has crossed the end
                    pad = np.zeros(length_per_sample - (sequence_length-ptr))
                    sequence_chop = np.concatenate((sequence_chop, pad))
                else:
                    sequence_chop = training_sequence[ptr:ptr+length_per_sample]
                
                ptr += step_sequence
                
                model_input = torch.tensor(sequence_chop, dtype=torch.float32).view(1, 1, len(sequence_chop)).to(device)
                model.to(device)
                
                model_output_timestep = model(model_input) # Getting model output

                sample_counter = int(ptr/step_sequence) - 1
                seq_model_output[sample_counter,:] = model_output_timestep[0] # Aggregating model output over all the samples to create a sequence output to be fed to the loss function
                
                counter += 1

        input_lengths = torch.tensor(n_samples)
        target_lengths = torch.tensor(len(target_sequence))

        loss = ctc_loss(seq_model_output, target_sequence, input_lengths, target_lengths)

        greedy_result = greedy_decoder(seq_model_output)
        greedy_transcript = " ".join(greedy_result)
        actual_transcript = get_actual_transcript(target_sequence)
        #greedy_wer = torchaudio.functional.edit_distance(actual_transcript, greedy_result) / len(actual_transcript)

        if greedy_transcript == actual_transcript:
            correct += 1

        total += 1  

        val_loss += loss.item()
        ### Implement val accuracy - need a decoder first

     
    val_loss /= len(X_val)
    val_accuracy = correct / total
    print(f"Epoch {epoch}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")




        
        
