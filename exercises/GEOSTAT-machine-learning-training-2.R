## ----general-options,echo=FALSE------------------------------------------

library(knitr)
# output code, but no warnings
opts_chunk$set(echo = TRUE,eval=TRUE,warning=FALSE)
# auto check dependencies (of cached chunks, its an approximation only)
opts_chunk$set(autodep = TRUE)
# dep_auto() # print dependencies 

setwd("/home/madlene/teaching/2017_GEOSTAT_machine-learning/exercises/")


## ----load-packages,message=FALSE-----------------------------------------
library(randomForest)
library(geoGAM)

## ----read-in-data--------------------------------------------------------
dim(berne)
# Continuous response 
d.ph10 <- berne[ berne$dataset == "calibration" & !is.na(berne$ph.0.10), ]
d.ph10 <- d.ph10[ complete.cases(d.ph10[13:ncol(d.ph10)]), ]
# covariates start at col 13
l.covar <- names(d.ph10[, 13:ncol(d.ph10)])

## ----fit-random-forest,cache=TRUE----------------------------------------
rf.ph <- randomForest(x = d.ph10[, l.covar],
                      y = d.ph10$ph.0.10)

## ----plot-covar-importance, fig.width=5, fig.height=6, fig.align='center', out.width='0.5\\textwidth', fig.cap="Covariate importance of 20 most important covariates for topsoil pH (before selection)."----
varImpPlot(rf.ph, n.var = 20, main = "")

## ----select-random-forest,cache=TRUE-------------------------------------
# speed up the process by removing 5-10 covariates at a time
s.seq <- sort( c( seq(5, 95, by = 5), 
                  seq(100, length(l.covar), by = 10) ), 
               decreasing = T)

# collect results in list
qrf.elim <- oob.mse <- list()

# save model and OOB error of current fit        
qrf.elim[[1]] <- rf.ph
oob.mse[[1]] <- tail(qrf.elim[[1]]$mse, n=1)
l.covar.sel <- l.covar

# Iterate through number of retained covariates           
for( ii in 1:length(s.seq) ){
  t.imp <- importance(qrf.elim[[ii]], type = 2)
  t.imp <- t.imp[ order(t.imp[,1], decreasing = T),]
  
  qrf.elim[[ii+1]] <- randomForest(x = d.ph10[, names(t.imp[1:s.seq[ii]])],
                                   y = d.ph10$ph.0.10 )
  oob.mse[[ii+1]] <- tail(qrf.elim[[ii+1]]$mse,n=1)
  
}


# Prepare a data frame for plot
elim.oob <- data.frame(elim.n = c(length(l.covar), s.seq[1:length(s.seq)]), 
                       elim.OOBe = unlist(oob.mse) )

## ----plot-selection-path,fig.align='center',echo=FALSE,fig.height = 5,out.width='0.8\\textwidth',fig.cap = "Path of out-of-bag mean squared error as covariates are removed. Minimum is found at 55 covariates."----

plot(elim.oob$elim.n, elim.oob$elim.OOBe, 
     ylab = "OOB error (MSE)",
     xlab = "n covariates", 
     pch = 20)
abline(v = elim.oob$elim.n[ which.min(elim.oob$elim.OOBe)], lty = "dotted")

## ----partial-dep-rf,fig.width=7,fig.height=8, fig.align='center', out.width='0.9\\textwidth',fig.cap = "Partial dependence plots for the 4 most important covariates."----
# select the model with minimum OOB error
rf.selected <- qrf.elim[[ which.min(elim.oob$elim.OOBe)]]

t.imp <- importance(rf.selected, type = 2)
t.imp <- t.imp[ order(t.imp[,1], decreasing = T),]

# 4 most important covariates
( t.3 <- names( t.imp[ 1:4 ] ) )

par( mfrow = c(2,2))

# Bug in partialPlot(): function does not allow a variable for the 
#  covariate name (e. g. x.var = name) in a loop
partialPlot(x = rf.selected, 
            pred.data = d.ph10[, names(rf.selected$forest$xlevels)], 
            x.var = "cl_mt_rr_3", ylab = "ph [-]", main = "") 
partialPlot(x = rf.selected, 
            pred.data = d.ph10[, names(rf.selected$forest$xlevels)], 
            x.var = "cl_mt_rr_11", ylab = "ph [-]", main = "" ) 
partialPlot(x = rf.selected, 
            pred.data = d.ph10[, names(rf.selected$forest$xlevels)], 
            x.var = "timeset", ylab = "ph [-]", main = "" ) 
partialPlot(x = rf.selected, 
            pred.data = d.ph10[, names(rf.selected$forest$xlevels)], 
            x.var = "cl_mt_rr_y", ylab = "ph [-]", main = "" ) 


## ----geogam,cache=TRUE---------------------------------------------------

# Fit a geoGAM model  
# only use 4 covariates to speed up
ph.geogam <- geoGAM(response = "ph.0.10",
                    covariates = c("timeset", "cl_mt_rr_3", 
                                   "tr_se_twi2m_s60", "sl_physio_neu"),
                    data = d.ph10,
                    max.stop = 20)

# select validation data
d.ph10.val <- berne[berne$dataset == "validation" & !is.na(berne$ph.0.10), ]
d.ph10.val <- d.ph10.val[complete.cases(d.ph10.val[13:ncol(d.ph10)]), ]

# compute predictions (mean) for each validation site
ph.pred <- predict(ph.geogam,
                   newdata = d.ph10.val)

## ----bootstrap-geogam,cache=TRUE-----------------------------------------
## compute model based bootstrap with 300 repetitions
# if this takes too long, reduce R to e.g. 50
ph.boot <- bootstrap(ph.geogam,
                     newdata = d.ph10.val,
                     R = 300)


## ----investigate-single-point,echo=FALSE,fig.pos='!h',fig.height=5,fig.width=4,fig.align='center', out.width='0.4\\textwidth',fig.cap= "Histogram of predictive distribution for one single prediction point, computed by 300 bootstrap repetitions."----

# plot predictive distribution for site in row 9
hist( as.numeric( ph.boot[ 9, ] ), 
      col = "grey", main = "",
      xlab = "predicted pH [-]", breaks = 12)

# compute 95 % prediction interval and add to plot
quant90 <- quantile( as.numeric( ph.boot[ 9, ] ), probs = c(0.05, 0.95) )
abline(v = c( quant90[1], quant90[2]), lty = "dashed")
abline(v = ph.pred[9], lty = "solid")


## ----create-intervall-plot,fig.height=5,fig.align='center',echo=FALSE, out.width='0.8\\textwidth',fig.cap= "Coverage of 90-prediction intervals computed by model-based boostrap."----

# compute 90% quantiles for each point
t.quant90 <- t( apply(ph.boot, 1, quantile, probs = c(0.05, 0.95) ) )

# get index for ranking in the plot
t.ix <- sort( ph.pred, index.return = T )$ix

# plot predictions in increasing order
plot(
  ph.pred[t.ix], type = "n",
  ylim = range(c(t.quant90, ph.pred, d.ph10.val$ph.0.10)),
  xlab = "rank of predictions", 
  ylab =  "ph [-]" 
) 

# add prediction intervals
segments(
  1:nrow( d.ph10.val ),
  t.lower <- (t.quant90[,1])[t.ix],
  1:nrow( d.ph10.val ),
  t.upper <- (t.quant90[,2])[t.ix],
  col = "grey"
)

# select colour for dots outside of intervals
t.col <- sapply(
  1:length( t.ix ),
  function( i, x, lower, upper ){
    as.integer( cut( x[i], c( -Inf, lower[i]-0.000001, 
                              x[i], upper[i]+0.000001, Inf ) ) )
  },
  x = d.ph10.val$ph.0.10[t.ix],
  lower = t.lower, upper = t.upper
)

# add observed values on top 
points(
  1:nrow( d.ph10.val ),
  d.ph10.val$ph.0.10[t.ix], cex = 0.7,
  pch = c( 16, 1, 16)[t.col],
  col = c( "darkgreen", "black", "darkgreen" )[t.col]
)
points(ph.pred[t.ix], pch = 16, cex = 0.6, col = "grey60")

# Add meaningfull legend
legend( "topleft", 
        bty = "n", cex = 0.85,
        pch = c( NA, 16, 1, 16 ), pt.cex = 0.6, lwd = 1,
        lty = c( 1, NA, NA, NA ), 
        col = c( "grey", "grey60", "black", "darkgreen" ), 
        seg.len = 0.8,
        legend = c(
          "90 %-prediction interval", 
          paste0("prediction (n = ", nrow(d.ph10.val), ")"),
          paste0("observation within interval (n = ", 
                 sum( t.col %in% c(2) ), ")" ),
          paste0("observation outside interval (n = ", 
                 sum( t.col %in% c(1,3)), ", ", 
                 round(sum(t.col %in% c(1,3)) / 
                         nrow(d.ph10.val)*100,1), "%)") )
)

## ----create-coverage-probabilty-plots,fig.align='center', out.width='0.55\\textwidth',fig.cap="Coverage probabilities of one-sided prediction intervals computed for the validation data set of topsoil pH of the Berne study area."----

# Coverage probabilities plot
# create sequence of nominal probabilities 
ss <- seq(0,1,0.01)
# compute coverage for sequence
t.prop.inside <- sapply(ss, function(ii){
  boot.quantile <-  t( apply(ph.boot, 1, quantile, probs = c(0,ii) ) )[,2]
  return( sum(boot.quantile <= d.ph10.val$ph.0.10)/nrow(d.ph10.val) )
})

plot(x = ss, y = t.prop.inside[length(ss):1], 
     type = "l", asp = 1,
     ylab = "coverage probabilities", 
     xlab="nominal probabilities" )
# add 1:1-line  
abline(0,1, lty = 2, col = "grey60")
# add lines of the two-sided 90 %-prediction interval
abline(v = c(0.05, 0.95), lty = "dotted", col = "grey20")

## ----session-info,results='asis'-----------------------------------------
toLatex(sessionInfo(), locale = FALSE)

## ----export-r-code,echo=FALSE--------------------------------------------
#purl("GEOSTAT-machine-learning-training-2.Rnw")

