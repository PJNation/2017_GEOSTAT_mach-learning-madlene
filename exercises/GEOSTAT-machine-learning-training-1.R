## ----general-options,echo=FALSE------------------------------------------

library(knitr)
# output code, but no warnings
opts_chunk$set(echo = TRUE,eval=TRUE,warning=FALSE)
# auto check dependencies (of cached chunks, its an approximation only)
opts_chunk$set(autodep = TRUE)
# dep_auto() # print dependencies 

setwd("/home/madlene/teaching/2017_GEOSTAT_machine-learning/exercises/")


## ----load-packages,message=FALSE-----------------------------------------
library(grpreg)
library(glmnet)
library(randomForest)
library(mboost)
library(geoGAM)
library(raster)

## ----read-in-data--------------------------------------------------------
dim(berne)
# Continuous response 
d.ph10 <- berne[berne$dataset == "calibration" & !is.na(berne$ph.0.10), ]
d.ph10 <- d.ph10[complete.cases(d.ph10[13:ncol(d.ph10)]), ]
# Binary response 
d.wlog100 <- berne[berne$dataset=="calibration"&!is.na(berne$waterlog.100), ]
d.wlog100 <- d.wlog100[complete.cases(d.wlog100[13:ncol(d.wlog100)]), ]
# Ordered/multinomial tesponse 
d.drain <- berne[berne$dataset == "calibration" & !is.na(berne$dclass), ]
d.drain <- d.drain[complete.cases(d.drain[13:ncol(d.drain)]), ]
# covariates start at col 13
l.covar <- names(d.ph10[, 13:ncol(d.ph10)])

## ----lasso-continuous-response,cache=TRUE--------------------------------

# define groups: dummy coding of a factor is treated as group
# find factors
l.factors <- names(d.ph10[l.covar])[ 
  t.f <- unlist( lapply(d.ph10[l.covar], is.factor) ) ]
l.numeric <-  names(t.f[ !t.f ])

g.groups <- c( 1:length(l.numeric), 
               unlist( 
                 sapply(1:length(l.factors), function(n){
                   rep(n+length(l.numeric), nlevels(d.ph10[, l.factors[n]])-1)
                 }) 
               ) 
)
# grpreg needs model matrix as input
XX <- model.matrix( ~., d.ph10[, c(l.factors, l.numeric), F])[,-1]

# cross validation (CV) to find lambda
ph.cvfit <- cv.grpreg(X = XX, y = d.ph10$ph.0.10, 
                      group = g.groups, 
                      penalty = "grLasso",
                      returnY = T) # access CV results


## ----lasso-predictions---------------------------------------------------

# choose optimal lambda: CV minimum error + 1 SE (see glmnet)
l.se <- ph.cvfit$cvse[ ph.cvfit$min ] + ph.cvfit$cve[ ph.cvfit$min ]
idx.se <- min( which( ph.cvfit$cve < l.se ) ) - 1

# select validation data
d.ph10.val <- berne[berne$dataset == "validation" & !is.na(berne$ph.0.10), ]
d.ph10.val <- d.ph10.val[complete.cases(d.ph10.val[13:ncol(d.ph10)]), ]

# create model matrix for validation set
newXX <- model.matrix( ~., d.ph10.val[, c(l.factors, l.numeric), F])[,-1]

t.pred.val <-  predict(ph.cvfit, X = newXX, 
                       type = "response",
                       lambda =  ph.cvfit$lambda[idx.se])

# get CV predictions, e.g. to compute R2
ph.lasso.cv.pred <- ph.cvfit$Y[,idx.se]

## ----lasso-get-model-----------------------------------------------------

## get all coefficients

# ph.cvfit$fit$beta[, idx.se ]

# only get the non-zero ones:
t.coef <- ph.cvfit$fit$beta[, idx.se ]
t.coef[ t.coef > 0 ]

## ----lasso-plot-cv,echo=FALSE,fig.width=7,fig.height=4.5, fig.align='center', out.width='0.8\\textwidth',fig.cap = "Cross validation error plotted against the tuning parameter lambda. The dashed line indicates lambda at minimal error, the dotted darkgrey line is the optimal lambda with minimal error + 1 SE."----

plot(ph.cvfit)
abline( h = l.se, col = "grey", lty = "dotted")
abline( v = log( ph.cvfit$lambda[ idx.se ]), col = "grey30", lty = "dotted")

## ----lasso-multinomial-response,cache = TRUE-----------------------------

# create model matrix for drainage classes
# use a subset of covariates only, because model optimization for 
# multinomial takes long otherwise
set.seed(42)
XX <- model.matrix(~.,d.drain[, l.covar[sample(1:length(l.covar), 20)]])[,-1]

drain.cvfit <- cv.glmnet( XX, d.drain$dclass, nfold = 10,  
                          keep = T, # access CV results
                          family = "multinomial", 
                          type.multinomial = "grouped")

## ----lasso-multinomial-response-coeffs,cache=TRUE------------------------

drain.fit <- glmnet( XX, d.drain$dclass,
                     family = "multinomial", 
                     type.multinomial = "grouped",
                     lambda = drain.cvfit$lambda.min)

# The coeffs are here:
# drain.fit$beta$well
# drain.fit$beta$moderate
# drain.fit$beta$poor

## ----glmboost,cache=TRUE-------------------------------------------------
# Fit model
ph.glmboost <- glmboost(ph.0.10 ~., data = d.ph10[ c("ph.0.10", l.covar)],
                        control = boost_control(mstop = 200),
                        center = TRUE)

# Find tuning parameter: mstop = number of boosting itertations
set.seed(42)
ph.glmboost.cv <- cvrisk(ph.glmboost, 
                         folds = mboost::cv(model.weights(ph.glmboost), 
                                            type = "kfold"))

# print optimal mstop
mstop(ph.glmboost.cv)

## print model with fitted coefficents 
# ph.glmboost[ mstop(ph.glmboost.cv)]

## ----glmboost-plot,fig.width=7,fig.height=5, fig.align='center', out.width='0.8\\textwidth',fig.cap = "Path of cross validation error along the boosting iterations.", echo = FALSE----
plot(ph.glmboost.cv)

## ----gamboost,cache=TRUE,message=FALSE-----------------------------------

# quick set up formula

# Response
f.resp <- "ph.0.10 ~ "

# Intercept, add to dataframe 
f.int <- "bols(int, intercept = F, df = 1)"
d.ph10$int <- rep(1, nrow(d.ph10))

# Smooth spatial surface (needs > 4 degrees of freedom)
f.spat <- "bspatial(x, y, df = 5, knots = 12)"

# Linear baselearners for factors, maybe use df = 5
f.fact <- paste( 
  paste( "bols(", l.factors, ", intercept = F)" ), 
  collapse = "+" 
)

# Splines baselearners for continuous covariates
f.num <- paste( 
  paste( "bbs(", l.numeric, ", center = T, df = 5)" ),
  collapse = "+"
)

# create complete formula 
ph.form <- as.formula( paste( f.resp, 
                              paste( c(f.int, f.num, f.spat, f.fact),
                                     collapse = "+")) ) 
# fit the boosting model
ph.gamboost  <- gamboost(ph.form, data = d.ph10,
                         control = boost_control(mstop = 200))

# Find tuning parameter
ph.gamboost.cv <- cvrisk(ph.gamboost, 
                         folds = mboost::cv(model.weights(ph.gamboost), 
                                            type = "kfold"))

## ----gamboost-results----------------------------------------------------
# print optimal mstop
mstop(ph.gamboost.cv)

## print model info 
ph.gamboost[ mstop(ph.glmboost.cv)]
## print number of chosen baselearners 
length( t.sel <-  summary( ph.gamboost[ mstop(ph.glmboost.cv)] )$selprob ) 

# Most often selected were: 
summary( ph.gamboost[ mstop(ph.glmboost.cv)] )$selprob[1:5]  

## ----gamboost-partial-plots,echo=FALSE,fig.width=7,fig.height=6, fig.align='center', out.width='0.8\\textwidth',fig.cap = "Residual plots of the 4 covariates with highest selection frequency."----
par(mfrow=c(2,2) )
plot(ph.gamboost[ mstop(ph.glmboost.cv)], which = names(t.sel[1:4]) )

## ----gamboost-partial-plots-spatial,echo=FALSE,fig.width=7,fig.height=5, fig.align='center', out.width='0.8\\textwidth',fig.cap = "Modelled smooth spatial surface based on the coordinates."----
par(mfrow=c(1,1) )
plot(ph.gamboost[ mstop(ph.glmboost.cv)], which = grep("bspat", names(t.sel), value = T) )

## ----geoGAM,cache=TRUE,message=FALSE-------------------------------------
ph.geogam <- geoGAM(response = "ph.0.10",
                    covariates = l.covar,
                    coords = c("x", "y"),
                    data = d.ph10, seed = 1)

## ----geoGAM-summary------------------------------------------------------

ph.geogam$gam.final$formula

##  plot summary and model selection information
# summary(ph.geogam)
# summary(ph.geogam, what = "path")

## ----geoGAM-map,echo = FALSE,fig.width=5,fig.height=5, fig.pos='!h',fig.align='center', out.width='0.5\\textwidth',fig.cap="GeogAM predictions for topsoil pH for the berne.grid section of the study area."----

# Create GRID output with predictions
sp.grid <- berne.grid[, c("x", "y")]
#add timeset - soil legacy data correction
# create prediction for newest set
berne.grid$timeset <- factor(rep("d1979_2010"), 
                             levels = levels(berne$timeset))
sp.grid$pred.ph.0.10 <- predict(ph.geogam, newdata = berne.grid)
# transform to sp object
coordinates(sp.grid) <- ~ x + y
# assign Swiss CH1903 / LV03 projection
proj4string(sp.grid) <- CRS("+init=epsg:21781")
# transform to grid
gridded(sp.grid) <- TRUE
plot(sp.grid)

## ----session-info,results='asis'-----------------------------------------
toLatex(sessionInfo(), locale = FALSE)

## ----export-r-code,echo=FALSE--------------------------------------------
#purl("GEOSTAT-machine-learning-training-1.Rnw")

