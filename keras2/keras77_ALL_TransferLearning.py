from tensorflow.keras.applications import VGG16, VGG19, Xception,\
    ResNet101,ResNet101V2,ResNet152,ResNet152V2,ResNet50,ResNet50V2,\
        InceptionV3,InceptionResNetV2,MobileNet,MobileNetV2,\
            DenseNet121,DenseNet169,DenseNet201,NASNetLarge,NASNetMobile,EfficientNetB0,EfficientNetB1

# model = VGG16()
# model = VGG19()
# model = Xception()
# model = ResNet101()
# model = ResNet101V2()
# model = ResNet152()
# model = ResNet152V2()
# model = ResNet50()
# model = ResNet50V2()
# model = InceptionV3()
# model = InceptionResNetV2()
# model = MobileNet()
# model = MobileNetV2()
# model = DenseNet121()
# model = DenseNet169()
# model = DenseNet201()
# model = NASNetLarge()
# model = NASNetMobile()
# model = EfficientNetB0()
model = EfficientNetB1()




model.trainable = False

model.summary()

print(len(model.weights))
print(len(model.trainable_weights))

# model = VGG16()
# Total params: 138,357,544
# Trainable params: 138,357,544
# Non-trainable params: 0
# 32
# 32

# model = VGG19()
# Total params: 143,667,240
# Trainable params: 0
# Non-trainable params: 143,667,240
# 38
# 0

# model = Xception()
# Total params: 22,910,480
# Trainable params: 0
# Non-trainable params: 22,910,480
# 236
# 0

# model = ResNet101()
# Total params: 44,707,176
# Trainable params: 0
# Non-trainable params: 44,707,176
# 626
# 0

# model = ResNet101V2()
# Total params: 44,675,560
# Trainable params: 0
# Non-trainable params: 44,675,560
# 544
# 0

# model = ResNet152()
# Total params: 60,419,944
# Trainable params: 0
# Non-trainable params: 60,419,944
# 932
# 0

# model = ResNet152V2()
# Total params: 60,380,648
# Trainable params: 0
# Non-trainable params: 60,380,648
# 816
# 0

# model = ResNet50()
# Total params: 25,636,712
# Trainable params: 0
# Non-trainable params: 25,636,712
# 320
# 0

# model = ResNet50V2()
# Total params: 25,613,800
# Trainable params: 0
# Non-trainable params: 25,613,800
# 272
# 0

# model = InceptionV3()
# Total params: 23,851,784
# Trainable params: 0
# Non-trainable params: 23,851,784
# 378
# 0

# model = InceptionResNetV2()
# Total params: 55,873,736
# Trainable params: 0
# Non-trainable params: 55,873,736
# 898
# 0

# model = MobileNet()
# Total params: 4,253,864
# Trainable params: 0
# Non-trainable params: 4,253,864
# 137
# 0

# model = MobileNetV2()
# Total params: 3,538,984
# Trainable params: 0
# Non-trainable params: 3,538,984
# 262
# 0

# model = DenseNet121()
# Total params: 8,062,504
# Trainable params: 0
# Non-trainable params: 8,062,504
# 606
# 0

# model = DenseNet169()
# Total params: 14,307,880
# Trainable params: 0
# Non-trainable params: 14,307,880
# 846
# 0

# model = DenseNet201()
# Total params: 20,242,984
# Trainable params: 0
# Non-trainable params: 20,242,984
# 1006
# 0

# model = NASNetLarge()
# Total params: 88,949,818
# Trainable params: 0
# Non-trainable params: 88,949,818
# 1546
# 0

# model = NASNetMobile()
# Total params: 5,326,716
# Trainable params: 0
# Non-trainable params: 5,326,716
# 1126
# 0

# model = EfficientNetB0()
# Total params: 5,330,571
# Trainable params: 0
# Non-trainable params: 5,330,571
# 314
# 0

# model = EfficientNetB1()
# Total params: 7,856,239
# Trainable params: 0
# Non-trainable params: 7,856,239
# 442
# 0