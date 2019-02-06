library(tuneR)
library(MASS)
library(keras)
power = 40
parseFolder = function(folderName) {
  records = array(0L, dim = c(0, power))
  files <- list.files(path=folderName, pattern="*.WAV", full.names=TRUE, recursive=FALSE)
  for( file in files) {
    wave = readWave(file)
    wave.mfcc = melfcc(wave, numcep = power)
    if(sum(is.nan(wave.mfcc)) > 0) {
      wave.mfcc = wave.mfcc[1:(nrow(wave.mfcc)-1),]
    }
    wave.means = colMeans(wave.mfcc)
    records = rbind(records, wave.means)
  }
  
  rownames(records) <- NULL
  data = as.data.frame(records)
  data = cbind(data, who = rep(folderName, nrow(data)))
  return(data)
}

load.single = function(file) {
  wave = readWave(file)
  wave.mfcc = melfcc(wave, numcep = power)
  wave.means = colMeans(wave.mfcc)
  wave.test = as.data.frame(matrix(wave.means, ncol = power, nrow = 1))
  return(wave.test)
}

load.test = function(folderName) {
  records = array(0L, dim = c(0, power+1))
  files <- list.files(path=folderName, pattern="*.WAV", full.names=TRUE, recursive=FALSE)
  for( file in files) {
    wave = readWave(file)
    wave.mfcc = melfcc(wave, numcep = power)
    wave.mfcc = wave.mfcc[complete.cases(wave.mfcc),]
    wave.means = colMeans(wave.mfcc)
    wave.means = c(wave.means, file)
    records = rbind(records, wave.means)
  }
  
  rownames(records) <- NULL
  data = as.data.frame(records)
  data = cbind(data, who = rep(folderName, nrow(data)))
  return(data)
}

model = keras_model_sequential()
model %>% 
  layer_dense(units = 256, activation = 'relu', input_shape = c(40)) %>% 
  layer_dropout(rate = 0.4) %>% 
  layer_dense(units = 128, activation = 'relu') %>%
  layer_dropout(rate = 0.3) %>%
  layer_dense(units = 5, activation = 'softmax')

model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = optimizer_rmsprop(),
  metrics = c('accuracy')
)


grita.data = parseFolder("grita")
bezi.data = parseFolder("bezi")
saturas.data = parseFolder("saturas")
mud.data = parseFolder("mud")
gomez.data = parseFolder("gomez")

all.data = rbind(grita.data, saturas.data, bezi.data, mud.data, gomez.data)
all.data = all.data[sample.int(nrow(all.data)),] 
y_train = to_categorical(as.numeric(all.data$who)-1)
x_train = as.matrix(all.data[,-ncol(all.data)])
history <- model %>% fit(
  x_train, y_train, 
  epochs = 30, batch_size = 5, 
  validation_split = 0.2
)
plot(history)
as.numeric(as.numeric(all.data$who)-1)
levels(all.data$who)


model.lda = lda(who ~ ., data = all.data)
all.data$who == predict(model.lda)$class

xardas.test = load.single("xardas.WAV")
bezi.test = load.single("bezi.WAV")
woman.test = load.single("kobieta.WAV")

test_x = load.test("test")
test_matrix = as.matrix(test_x[,1:40])
test_x

model %>% predict_classes(as.matrix(test_matrix)) -> predictions
levels(all.data$who)[predictions+1]

predict(model.lda, xardas.test)
predict(model.lda, bezi.test)
predict(model.lda, woman.test)

set.seed(13)
separations = kmeans(all.data[,1:power], 5)$cluster
separations
all.data[,ncol(all.data)]
k.groups = as.factor(c("mud","bezi","grita","saturas", "gomez")[separations])
all.data[,ncol(all.data)] == k.groups
