
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
from utils import get_actual_transcript, get_model_savepath
import torchaudio
import datetime

# Tensorboard
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

device = torch.device("cuda:0")

labels_int = np.arange(11).tolist()
labels = [f"{i}" for i in labels_int] # Tokens to be fed into greedy decoder
greedy_decoder = GreedyCTCDecoder(labels=labels)

model_save_path = get_model_savepath()
model_save_iterations = 20

# Model Parameters
input_size = 1  # Number of input channels
hidden_size = 128
num_layers = 3
output_size = 11  # Number of output classes
dropout_rate = 0.2

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
step_sequence = 100
window_overlap = 50
length_per_sample = 150

epochs = 20

# Add over epochs
for epoch in range(epochs):

    #################### Training Loop #################
    print(f"Epoch {epoch}")
    model.train()
    for i in tqdm(range(len(X_train))):

        training_sequence, target_sequence = X_train[i], torch.tensor(y_train[i]).to(device)

        sequence_length = len(training_sequence)
        
        n_samples = math.ceil(sequence_length/step_sequence) # Since we send the last one even if it is small as can be

        seq_model_output = torch.zeros(n_samples, n_classes+1) # To include the blank token

        ptr = 0
        counter = 0
        sequences = torch.zeros([n_samples, 1, length_per_sample])
        while ptr <= sequence_length:
            
            try:
                if ptr + length_per_sample > sequence_length:
                    sequence_chop = training_sequence[ptr:-1] # For when the window has crossed the end
                    pad = np.zeros(length_per_sample - (sequence_length-ptr) + 1)
                    sequence_chop = np.concatenate((sequence_chop, pad)).tolist()
                else:
                    sequence_chop = training_sequence[ptr:ptr+length_per_sample]
                
                sequence_chop = torch.tensor(sequence_chop, dtype=torch.float32).view(1, len(sequence_chop))

                sequences[counter] = sequence_chop
            
            except Exception as e:
                print(e)
            
            ptr += step_sequence
            counter+=1
                
       
        model.to(device)

        # Zero out the gradients
        optimizer.zero_grad()
            
        model_output_timestep = model(sequences) # Getting model output

        counter += 1

        input_lengths = torch.tensor(n_samples)
        target_lengths = torch.tensor(len(target_sequence))

        loss = ctc_loss(model_output_timestep, target_sequence, input_lengths, target_lengths)
        
        loss.backward()

        # Update the weights
        optimizer.step()

        if i % 100 == 0:
            print(f"\nEpoch {epoch} Batch {i}")
            print(f"Loss {loss.item()}")
            greedy_result = greedy_decoder(model_output_timestep)
            greedy_transcript = " ".join(greedy_result)
            actual_transcript = get_actual_transcript(target_sequence)
            motif_err = torchaudio.functional.edit_distance(actual_transcript, greedy_result) / len(actual_transcript)

            print(f"Transcript: {greedy_transcript}")
            print(f"Actual Transcript: {actual_transcript}")
            print(f"Motif Error Rate: {motif_err}")

            writer.add_scalar("Loss/train", loss, epoch)
            writer.add_scalar("Err/Train", motif_err, epoch)
        
        # Saving model weights
        if i % model_save_iterations == 0:
            torch.save({
            'epoch': epoch,
            'batch':i,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, model_save_path)

    
    ################## Validation Loop ####################
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for i in tqdm(range(len(X_val))):

            training_sequence, target_sequence = X_val[i], torch.tensor(y_val[i]).to(device)

            sequence_length = len(training_sequence)
            
            n_samples = math.ceil(sequence_length/step_sequence) # Since we send the last one even if it is small as can be

            seq_model_output = torch.zeros(n_samples, n_classes+1) # To include the blank token

            ptr = 0
            counter = 0
            sequences = torch.zeros([n_samples, 1, length_per_sample])
            while ptr <= sequence_length:
                
                try:
                    if ptr + length_per_sample > sequence_length:
                        sequence_chop = training_sequence[ptr:-1] # For when the window has crossed the end
                        pad = np.zeros(length_per_sample - (sequence_length-ptr) + 1)
                        sequence_chop = np.concatenate((sequence_chop, pad)).tolist()
                    else:
                        sequence_chop = training_sequence[ptr:ptr+length_per_sample]
                    
                    sequence_chop = torch.tensor(sequence_chop, dtype=torch.float32).view(1, len(sequence_chop))

                    sequences[counter] = sequence_chop
                
                except Exception as e:
                    print(e)
                
                ptr += step_sequence
                counter+=1
                    
        
            model.to(device)

            # Zero out the gradients
            optimizer.zero_grad()
                
            model_output_timestep = model(sequences) # Getting model output

            counter += 1

            input_lengths = torch.tensor(n_samples)
            target_lengths = torch.tensor(len(target_sequence))

            loss = ctc_loss(model_output_timestep, target_sequence, input_lengths, target_lengths)

            if greedy_transcript == actual_transcript:
                correct += 1

            total += 1  
            val_loss += loss.item()

    val_loss /= len(X_val)
    val_accuracy = correct / total
    print(f"Epoch {epoch}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")
    writer.add_scalar("Loss/validation", val_loss, epoch)




        
        
