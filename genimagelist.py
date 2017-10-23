num_files = 17
with open('imagelist.txt', 'w+') as f:
    for i in range(num_files):
        f.write(str(i+1) + "-1.jpg")
        f.write("\n")
        f.write(str(i+1) + "-2.jpg")
        f.write("\n")
        f.write(str(i+1) + "-3.jpg")
        if i < num_files - 1:
            f.write("\n")