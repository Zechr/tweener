import sys
num_files = int(sys.argv[1])
with open('imagelist.txt', 'w+') as f, open('imagelist2.txt', 'w+') as f2:
    for i in range(num_files):
        f.write("images/" + str(i+1) + "-1.jpg")
        f.write("\n")
        f.write("images/" + str(i+1) + "-2.jpg")
        f.write("\n")
        f.write("images/" + str(i+1) + "-3.jpg")
        if i < num_files - 1:
            f.write("\n")
        f2.write("images/" + str(i+1) + ".png")
        f2.write("\n")
        f2.write("images/" + str(i+1) + "-1.jpg")
        f2.write("\n")
        f2.write("images/" + str(i+1) + "-2.jpg")
        if i < num_files - 1:
            f2.write("\n")