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

model = keras_model_sequential()
model %>% 
  layer_dense(units = 256, activation = 'relu', input_shape = c(40)) %>% 
  layer_dropout(rate = 0.4) %>% 
  layer_dense(units = 128, activation = 'relu') %>%
  layer_dropout(rate = 0.3) %>%
  layer_dense(units = 5, activation = 'softmax')


grita.data = parseFolder("grita")
bezi.data = parseFolder("bezi")
saturas.data = parseFolder("saturas")
mud.data = parseFolder("mud")
gomez.data = parseFolder("gomez")

all.data = rbind(grita.data, saturas.data, bezi.data, mud.data, gomez.data)
all.data = all.data[sample.int(nrow(all.data)),] 



model.lda = lda(who ~ ., data = all.data)
all.data$who == predict(model.lda)$class

xardas.test = load.single("xardas.WAV")
bezi.test = load.single("bezi.WAV")
woman.test = load.single("kobieta.WAV")

predict(model.lda, xardas.test)
predict(model.lda, bezi.test)
predict(model.lda, woman.test)

set.seed(13)
separations = kmeans(all.data[,1:power], 5)$cluster
separations
all.data[,ncol(all.data)]
k.groups = as.factor(c("mud","bezi","grita","saturas", "gomez")[separations])
all.data[,ncol(all.data)] == k.groups
