myscale <- function(x) {
  xscaled <- apply(x, 2, function(y) (y-mean(y))/sd(y))
  return(xscaled)
}
# a function that gives knn classification results
myfun <- function(testrow,train1,k,ytrain1) {
  xdist <- apply(train1, 1, function(xrow) sum((xrow-testrow)^2))
  nbs <- order(xdist)[1:k]
  return(round(mean(ytrain1[nbs])))
}

knnclass <- function(xtrain, xtest, ytrain) {
  set.seed(1)
  # rescale
  xtest <- sapply(1:ncol(xtest),function(i) {
    xtest[,i] <- (xtest[,i]-mean(xtrain[,i]))/sd(xtrain[,i])
  })  
  xtest <- as.data.frame(xtest)
  xtrain <- myscale(xtrain);
  # split the training set
  w <- sample(1:nrow(xtrain), nrow(xtrain)*0.8, replace = FALSE)
  train1 <- xtrain[w,]; ytrain1 <- ytrain[w]
  valid <- xtrain[-w,]; yvalid <- ytrain[-w]
  # choose optimal k
  miserror <- rep(0,14)
  for ( k in 1:14){
    yhat <- apply (valid, 1, myfun, train1=train1,k=k+1,ytrain1=ytrain1)
    miserror[k] <- mean(yhat != yvalid) # misclassification error
  }
  kbest <- which.min(miserror)+1
  yhat2 <- apply(xtest, 1, myfun, train1=xtrain,k=kbest,ytrain1=ytrain)
  return(yhat2)
}
