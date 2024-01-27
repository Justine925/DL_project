import os
import glob
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

 
### MAPS ###
# Mapping for species and breeds
species_map = {
    0: "Cat",
    1: "Dog"
}
breeds_map = {
    0: "Abyssinian",
    1: "american_bulldog",
    2: "american pit bull terrier",
    3: "basset hound",
    4: "beagle",
    5: "Bengal",
    6: "Birman",
    7: "Bombay",
    8: "boxer",
    9: "British shorthair",
    10: "chihuahua",
    11: "Egyptian_Mau",
    12: "english cocker spaniel",
    13: "english setter",
    14: "german shorthaired",
    15: "great pyrenees",
    16: "havanese",
    17: "japanese chin",
    18: "keeshond",
    19: "leonberger",
    20: "Maine coon",
    21: "miniature pinsher",
    22: "newfoundland",
    23: "Persian",
    24: "pomeranian",
    25: "pug",
    26: "Ragdoll",
    27: "Russian blue",
    28: "saint bernard",
    29: "samoyed",
    30: "scottish terrier",
    31: "shiba inu",
    32: "Siamese",
    33: "Sphynx",
    34: "staffordshire bull terrier",
    35: "wheaten terrier",
    36: "yorkshire terrier",
}

### IMAGES ###

def pil_loader(path):
    """Load an image from the given file path using PIL and convert it to RGB format"""
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def extract_species_and_number_img(img_path):
    """Extract species and image number from the image file name"""
    img_name = os.path.basename(img_path)
    species = img_name.rsplit('_', 1)[0].lower() 
    number = int(img_name.rsplit('_', 1)[1].split('.')[0])
    return (species, number)

class ImageDataset(Dataset):
    """PyTorch dataset for loading images"""
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        # Get a sorted list of image files using a custom key function
        self.file_list = sorted(glob.glob(self.root_dir + "*.jpg"), key=extract_species_and_number_img )

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_name = self.file_list[idx]
        image = pil_loader(img_name)
        if self.transform:
            image = self.transform(image)
        return image
    
    def getname(self,idx):
        return self.file_list[idx]
    
### ANNOTATIONS ###

def extract_species_and_number_annot(line):
    """Extract species and image number from an annotation line"""
    parts = line.split()
    species = parts[0].rsplit('_', 1)[0].lower()  # Extract name
    number = int(parts[0].rsplit('_', 1)[1])  # Extract number
    return (species, number)

def getAnnotLists():
    """Read annotations from a file and return lists of image names, class IDs, and species IDs"""
    image_names = []
    class_ids = []
    species_ids = []
    breed_ids = []
    with open('C:\\Users\\Justine_B\\OneDrive\\Documents\\UTC\\GI05\\DL\\DeepLearningAndGenerativeModelsCourse-main\\Project\\Extracted_annotations\\annotations\\list.txt', 'r') as file:
        lines = [line.strip() for line in file if not line.startswith("#")]
        # Sort the lines based on the species and number using a custom key function
        sorted_lines = sorted(lines, key=extract_species_and_number_annot)
        for line in sorted_lines:
            columns = line.split()
            image_names.append(columns[0])
            # Subtract 1 to convert to zero-based index
            class_ids.append((int(columns[1])-1))
            species_ids.append((int(columns[2])-1))
    return image_names,class_ids,species_ids

### SETS ###

def getSetsIndices(data_imgs,class_ids):
    """Generate indices for training and testing sets"""
    train_indices = []
    test_indices = []
    i = 0
    for c in range(37):
        #For each class, arround 80% go in the training set and the rest in the test set.
        nb_c = class_ids.count(c)
        train_nb_c = (nb_c//10)*8
        for k in range(nb_c):
            if k < train_nb_c:
                train_indices += [i]
            else :
                test_indices += [i]
            i += 1 
    #Randomizes the order of clues in each set
    np.random.shuffle(train_indices)
    np.random.shuffle(test_indices)
    return train_indices, test_indices

### RESULTS ###

def print_accuracy_results(total,correct):
    """Print the accuracy of the network on the test images"""
    print(f'Accuracy of the network on the test images : {100 * correct // total} %')

def print_accuracy_species_results(conf_matrix):
    """Print the accuracy of the network for each class"""
    print("Accuracy of the network for each class :")
    tot_class = np.sum(conf_matrix, axis=0)
    for i in range(37):
        class_correct = conf_matrix[i][i]
        class_total = tot_class[i]
        class_accuracy = 100 * class_correct / class_total
        print(f'Accuracy for class {i} {breeds_map[i]}: {class_accuracy:.2f}%')
    
def print_confusion_matrix(conf_matrix):
    """Print the confusion matrix"""
    print("Confusion Matrix :")
    print()
    print("   ", '  '.join([str(i) for i in range(0, 11)]),' '.join([str(i) for i in range(11,37)]))
    for i in range(37):
        if i < 10:
            print(i,end ="  ")
        else :
            print(i,end=" ")
        for elem in conf_matrix[i]:
            if elem < 10 :
                print(" ",int(elem), end="  ")
            else :
                print(int(elem), end=" ")
        print()

def get_breeds_species(i):
    """Get the species (Cat/Dog) based on breed index"""
    breed = breeds_map[i]
    if breed[0].isupper():
        return 0
    return 1

def print_classification_errors(conf_matrix):
    """Print the classification errors confusion matrix. Gives the number
    of cats/dogs misclassified in another breed of cat/dog """
    errors_conf_matrix = np.zeros((2,2))
    for i in range(37):
        species_i = get_breeds_species(i)
        for j in range(37):
            if j != i:
                species_j = get_breeds_species(j)
                errors_conf_matrix[species_i,species_j] += conf_matrix[i,j]
    print("Classification Errors Confusion Matrix :")
    print()
    print("    ", 0," ",1)
    for i in range(2):
        print(i,end ="  ")
        for elem in errors_conf_matrix[i]:
            if elem < 10 :
                print(" ",int(elem), end=" ")
            elif elem < 100 :
                print("",int(elem), end=" ")
            else :
                print(int(elem), end=" ")
        print()

def write_result_report(doc_name,total,correct, conf_matrix):
    with open("./results/"+doc_name, 'w') as f:
            #Global accuracy
            f.write(f'Accuracy of the network on the test images : {100 * correct // total} %\n')
            f.write('\n')
            #Breeds accuracy
            f.write("Accuracy of the network for each class :\n")
            tot_class = np.sum(conf_matrix, axis=0)
            for i in range(37):
                class_correct = conf_matrix[i][i]
                class_total = tot_class[i]
                class_accuracy = 100 * class_correct / class_total
                f.write(f'Accuracy for class {i} {breeds_map[i]}: {class_accuracy:.2f}%\n')
            f.write('\n')
            #Confusion matrix
            f.write("Confusion Matrix :\n")
            f.write('\n')
            f.write(" ")
            for i in range(11):
                f.write( '   '+str(i))
            for i in range (11,37):
                f.write( ' '+str(i)+" ")
            f.write('\n')
            for i in range(37):
                if i < 10:
                    f.write(str(i)+"  ")
                else :
                    f.write(str(i)+" ")
                for elem in conf_matrix[i]:
                    if elem < 10 :
                        f.write(" "+str(int(elem))+"  ")
                    else :
                        f.write(str(int(elem))+"  ")
                f.write('\n')
            f.write('\n')
            #Error confusion matrix
            errors_conf_matrix = np.zeros((2,2))
            for i in range(37):
                species_i = get_breeds_species(i)
                for j in range(37):
                    if j != i:
                        species_j = get_breeds_species(j)
                        errors_conf_matrix[species_i,species_j] += conf_matrix[i,j]
            f.write("Classification Errors Confusion Matrix :")
            f.write('\n')
            f.write("     "+str(0)+"   "+str(1)+"\n")
            for i in range(2):
                f.write(str(i)+"  ")
                for elem in errors_conf_matrix[i]:
                    if elem < 10 :
                        f.write("  "+str(int(elem))+" ")
                    elif elem < 100 :
                        f.write(" "+str(int(elem))+" ")
                    else :
                        f.write(str(int(elem))+" ")
                f.write('\n')
    f.close()


