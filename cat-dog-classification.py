
#CNN Mimarimiz
from keras.models import Sequential

#Convolution +ReLU
from keras.layers import Conv2D

#Max Pooling
from keras.layers import MaxPooling2D

#Flatten
from keras.layers import Flatten

#Fully Connected --> fully connected layer oluşturmak için kullanılır
from keras.layers import Dense

#Dropout ile bazı nöronları çıkarmış oluyoruz. ÖZelliklerinin birbirine bağımlı olmasının önüne geçiyoruz
from keras.layers import Dropout

#CNN i başlat
siniflandirici=Sequential()

#CNN layer ekleme
siniflandirici.add(Conv2D(32,(3,3),
                          input_shape=(64,64,3),
                          activation='relu'))
"""
32 --> filtre sayısı
(3,3) --> filtre büyüklüğü
input shape --> CNN e gelmesini beklediğimiz resimlerin büyüklüğü
64,64,3 --> 64,64 boyut --> 3 derinlik RGB formatı
"""

#Max Pooling uygulama
siniflandirici.add(MaxPooling2D(2,2)) #Resmin boyutunda yarı yarıya azalma olacak

#İkinci CNN katmanı ekleme

siniflandirici.add(Conv2D(32, (3,3),  activation = "relu"))
siniflandirici.add(MaxPooling2D(2,2))

siniflandirici.add(Conv2D(64, (3,3),  activation = "relu"))
siniflandirici.add(MaxPooling2D(2,2))


siniflandirici.add(Flatten()) #Bu katmandaki parametre sayısında herhangi bir değişiklik yok. Öğrenmesi gereken parametre sayısı sıfır.

siniflandirici.add(Dense(units=128,activation='relu'))

siniflandirici.add(Dropout(0.5))

siniflandirici.add(Dense(units=1, activation='sigmoid'))

siniflandirici.compile(optimizer='adam',loss='binary_crossentropy', metrics=['accuracy'])

siniflandirici.summary()

"""
total params : bu ağdaki toplam parametre sayısı
trainable params : eğitilebilir parametre sayısı
non-trainable params : eğitilemez parametre sayısı
"""

# Data Augmentation --> elimdeki veri sayısını artırma yöntemleri
from keras.preprocessing.image import ImageDataGenerator

#eğitim veri setine uygulayacağım değer
train_datagen=ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
    )

#test veri setine uygulayacağım değer
#bunu eğitim için yapıyorsam test için de yapmam gerekiyor --> rescale
#çünkü test verisini artırmak gibi bir amacı yok
test_datagen=ImageDataGenerator(
    rescale=1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set', #dosya yolu
                                           target_size = (64,64), #tüm resimler 64 64 boyutuna dönüştürülecek
                                           batch_size = 32, #her 32 adımda bir filtreleri güncelle
                                           #her turda kaç adım atmam gerektiği
                                           class_mode = 'binary')
#flow_from_directory --> klasörlerin içerisinden bunları alıcam ve benim belirlediğim hedefe getirecek

test_set=test_datagen.flow_from_directory('dataset/test_set',
                                          target_size=(64,64),
                                          batch_size=32,
                                          class_mode='binary')



siniflandirici.fit_generator(training_set, #eğitim seti
                             steps_per_epoch = 8000//32, #her adımda(step) uygulanacak epoch size
                             epochs = 15,
                             validation_data = test_set,
                             validation_steps = 2000//32)

#Ağırlıkları kaydetme
siniflandirici.save_weights('agirliklar_2022.h5')

siniflandirici.load_weights('agirliklar_2022.h5')

siniflandirici.save('model2022')

from keras.models import load_model
siniflandirici2=load_model(('model2022'))


































