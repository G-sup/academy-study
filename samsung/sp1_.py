import numpy as np





a = np.array([range(1,11),range(11,21),range(21,31)])

a = a.T
print(a.shape)

def split_x(seq,size,cols):
    x , y = [] , []
    for i in range(len(seq)):
        x_end_number = i + size
        y_end_number = x_end_number + cols -1
        if y_end_number > len(seq) :
            break
        tem_x = seq[i : x_end_number]
        tem_y = seq[x_end_number-1:y_end_number , -1]
        x.append(tem_x)
        y.append(tem_y)
    return np.array(x),np.array(y)
    
x, y = split_x(a,3,1)

print(x)
print(y)



print(x.shape)
print(y.shape)





# import numpy as np








# def split_x(seq, size):
#     aaa = []
#     for i in range(len(seq) - size + 1):
#         subset = seq[i : ( i + size )]
#         aaa.append(subset) #[item for item in subset]
#     print(type(aaa))
#     return np.array(aaa)

# size = 4

# dataset = split_x(a,size)

# # print(dataset)

# x = dataset[:, :4] # x = dataset[0:6, 0:4]
# y = dataset[0:6, -1] # y = dataset[0:6, 4], y = dataset[:, 4]

# print(x)
# print(y)









# import numpy as np



# a = np.array([1,2,3,4,5,6,7,8,9,10])

# def split_x(a,size):
#     x , y = list() , list()
#     for i in range(len(a)):
#         end_number = i + size
#         if end_number > len(a) - 1:
#             break
#         tem_x, tem_y = a[i:end_number],a[end_number]
#         x.append(tem_x)
#         y.append(tem_y)
#     return np.array(x),np.array(y)
    
# x, y = split_x(a,4)

# print(x)
# print(y)