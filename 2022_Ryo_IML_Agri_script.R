######################################################################################
#
#  Interpretable ML analysis with the example dataset 
#    Su et al. (2021) A global dataset for crop production under conventional tillage and no tillage systems
#    in Scientific Data
#    https://www.nature.com/articles/s41597-021-00817-x
#
#                                    21.08.2022 Masahiro Ryo
#
######################################################################################

# library
library(tidyverse)
library(stars)
library(rnaturalearth)
library(patchwork)

library(caret)
library(pdp)
library(vip)
library(iml)


# processing data
df <- read.csv("../Database.csv", stringsAsFactors=T) %>% na.omit
df$NT_effect <- df$NT_effect %>% fct_rev()
df <- df[-which(df$Yield_change > quantile(df$Yield_change, 0.975)),] # remove extreme values

# Only for maize
df <- df %>% filter(Crop=="maize") %>% dplyr::select(-Yield_NT,-NT_effect, -Crop)

# world map with number of factors
world <- ne_countries(scale="small", returnclass = "sf") 
map_global_star <- world$Cumulative %>% st_as_stars()


world_map <-
  ggplot(data = world) +
  geom_sf() +
  geom_point(data=df, aes(x=Longitude, y=Latitude), size=1, color="black")+
  theme_bw() 

world_hist <-
ggplot(data=df, aes(y=Yield_change) ) +
  geom_histogram() +
  coord_flip() + 
  ylab("Relative change in crop yield") +
  theme_bw() 

Fig01 <- (world_map / world_hist) + 
  plot_layout(ncol=2, widths = c(2,1), heights = c(1,1)) + 
  plot_annotation(tag_levels = 'a')

Fig01

# Split data for machine learning -----------------------------------------

set.seed(42)
train_test_split <- sample(c(1:nrow(df)), 0.8*nrow(df), replace=F) 
data_train <- df[train_test_split,]
data_test  <- setdiff(df, data_train)


# Machine learning algorithm implementation -------------------------------


tc = trainControl(method = "cv", number = 5) # caret

set.seed(123)
model.lm   = caret::train(Yield_change ~., data=data_train, method="glmStepAIC", trControl=tc)
set.seed(123)
model.cart = caret::train(Yield_change ~., data=data_train, method="ctree", trControl=tc)
set.seed(123)
model.rf   = caret::train(Yield_change ~., data=data_train, method="rf", trControl=tc)
set.seed(123)
model.gbm  = caret::train(Yield_change ~., data=data_train, method="gbm", trControl=tc,
                          tuneGrid=expand.grid(n.trees=(1:5)*500, interaction.depth=(1:5)*3,
                                               shrinkage=0.1, n.minobsinnode=10)) 



# -------------------------------------------------------
# performance evaluation

# with data_test
pred.lm   <- predict(model.lm, data_test)
pred.cart <- predict(model.cart, data_test)
pred.rf   <- predict(model.rf, data_test)
pred.gbm  <- predict(model.gbm, data_test)


# r2: obs vs pred
r2.lm   <- cor(pred.lm, data_test$Yield_change)^2 %>% round(.,4)
r2.cart <- cor(pred.cart, data_test$Yield_change)^2 %>% round(.,4)
r2.rf   <- cor(pred.rf, data_test$Yield_change)^2 %>% round(.,4)
r2.gbm  <- cor(pred.gbm, data_test$Yield_change)^2 %>% round(.,4)

r2 <- data.frame(r2 = c(r2.lm,r2.cart,r2.rf,r2.gbm), algorithm = c("Linear model","Decision Tree","Random Forests","Gradient Boosting") %>% factor(.,levels=c("Linear model","Decision Tree","Random Forests","Gradient Boosting")))

Fig02 <-
ggplot(r2, aes(x=algorithm, y=r2, fill=algorithm)) + 
  geom_bar(stat="identity") + 
  ylab("R-squred") +
  scale_fill_manual(values=c("Linear model"="grey",
                             "Decision Tree"="orange",
                             "Random Forests"="darkgreen",
                             "Gradient Boosting"="darkblue")) +
  theme_bw() +
  theme(legend.position = "none")

Fig02

#-----------------------------------------------------------
# Interpretable machine learning 

# variable importance (feature importance)

# model agnostic post-hoc interpretability
# permutation-based feature importance
set.seed(123)
pvip_lm <- vip(model.lm, method="permute", train=data_train, target="Yield_change", metric="rsquared", 
               pred_wrapper=predict, nsim=30, geom="boxplot",aesthetics = list(fill = "grey", color="black")) + labs(title="Linear model") +theme_bw()
set.seed(123)
pvip_cart <- vip(model.cart, method="permute", train=data_train, target="Yield_change", metric="rsquared", 
                 pred_wrapper=predict, nsim=30, geom="boxplot",aesthetics = list(fill = "orange", color="black")) + labs(title="Decision Tree") +theme_bw()
set.seed(123)
pvip_rf <- vip(model.rf, method="permute", train=data_train, target="Yield_change", metric="rsquared", 
               pred_wrapper=predict, nsim=30, geom="boxplot",aesthetics = list(fill = "darkgreen", color="black")) + labs(title="Random Forests") +theme_bw()
set.seed(123)
pvip_gbm <- vip(model.gbm, method="permute", train=data_train, target="Yield_change", metric="rsquared", 
                pred_wrapper=predict, nsim=30, geom="boxplot",aesthetics = list(fill = "darkblue", color="black")) + labs(title="Gradient Boosting") +theme_bw()

# patchwork

plot_pvip_all <-
  pvip_lm + pvip_cart + pvip_rf + pvip_gbm

Fig03 <-
plot_pvip_all + 
  plot_annotation(tag_levels = 'a')

Fig03

# Friedman's H-index
# VI-based interaction statistic
int.lm <- vint(
  object = model.lm,                    # fitted model object
  feature_names = c("Yield_CT","ST","Soil_cover_NT", "Tmax", "Tave", "Tmin"),  # features for which to compute pairwise interactions statistics
  parallel = TRUE
)
int.cart <- vint(
  object = model.cart,                    # fitted model object
  feature_names = c("Yield_CT","ST","Soil_cover_NT", "Tmax", "Tave", "Tmin"),  # features for which to compute pairwise interactions statistics
  parallel = TRUE
)
int.rf <- vint(
  object = model.rf,                    # fitted model object
  feature_names = c("Yield_CT","ST","Soil_cover_NT", "Tmax", "Tave", "Tmin"),  # features for which to compute pairwise interactions statistics
  parallel = TRUE
)
int.gbm <- vint(
  object = model.gbm,                    # fitted model object
  feature_names = c("Yield_CT","ST","Soil_cover_NT", "Tmax", "Tave", "Tmin"),  # features for which to compute pairwise interactions statistics
  parallel = TRUE
)

interaction_strength.lm <-
ggplot(int.lm[1:10, ], aes(reorder(Variables, Interaction), Interaction)) +
  geom_bar(stat="identity", fill="grey") +
  labs(x = "", y = "Interaction strength") +
  theme(axis.text.x = element_text(angle = 45, vjust = 1)) +
  ylim(c(0,0.01))+
  coord_flip()  +
  theme_bw() + labs(title="Linear model")

interaction_strength.cart <-
  ggplot(int.cart[1:10, ], aes(reorder(Variables, Interaction), Interaction)) +
  geom_bar(stat="identity", fill="orange") +
  labs(x = "", y = "Interaction strength") +
  theme(axis.text.x = element_text(angle = 45, vjust = 1)) +
  coord_flip()  +
  theme_bw() + labs(title="Decision Tree")

interaction_strength.rf <-
  ggplot(int.rf[1:10, ], aes(reorder(Variables, Interaction), Interaction)) +
  geom_bar(stat="identity", fill="darkgreen") +
  labs(x = "", y = "Interaction strength") +
  theme(axis.text.x = element_text(angle = 45, vjust = 1)) +
  coord_flip()  +
  theme_bw() + labs(title="Random Forests")

interaction_strength.gbm <-
  ggplot(int.gbm[1:10, ], aes(reorder(Variables, Interaction), Interaction)) +
  geom_bar(stat="identity", fill="darkblue") +
  labs(x = "", y = "Interaction strength") +
  theme(axis.text.x = element_text(angle = 45, vjust = 1)) +
  coord_flip()  +
  theme_bw() + labs(title="Gradient Boosting")

interaction_strength <-
  interaction_strength.lm + interaction_strength.cart +
  interaction_strength.rf + interaction_strength.gbm

Fig04 <-
interaction_strength + 
  plot_annotation(tag_levels = 'a')
  
Fig04

# -----------------------------------------------------------------------
# partial dependence plot

# Yield_CT: The most important variable
pdp_Yield_CT   <- rbind(
  model.lm %>%  partial(pred.var=c("Yield_CT")) %>% cbind(., algorithm = "Linear model"),
  model.cart %>%  partial(pred.var=c("Yield_CT"), approx=T) %>% cbind(., algorithm = "Decision Tree"),
  model.rf %>%  partial(pred.var=c("Yield_CT"), approx=T) %>% cbind(., algorithm = "Random Forests"),
  model.gbm %>%  partial(pred.var=c("Yield_CT"), approx=T)  %>% cbind(., algorithm = "Gradient Boosting")
) 
pdp_Yield_CT$algorithm <- factor(pdp_Yield_CT$algorithm, levels=c("Linear model","Decision Tree","Random Forests","Gradient Boosting"))

# Tmax: The 2nd most important variable
pdp_Tmax   <- rbind(
  model.lm %>%  partial(pred.var=c("Tmax")) %>% cbind(., algorithm = "Linear model"),
  model.cart %>%  partial(pred.var=c("Tmax"), approx=T) %>% cbind(., algorithm = "Decision Tree"),
  model.rf %>%  partial(pred.var=c("Tmax"), approx=T) %>% cbind(., algorithm = "Random Forests"),
  model.gbm %>%  partial(pred.var=c("Tmax"), approx=T)  %>% cbind(., algorithm = "Gradient Boosting")
) 
pdp_Tmax$algorithm <- factor(pdp_Tmax$algorithm, levels=c("Linear model","Decision Tree","Random Forests","Gradient Boosting"))


Fig05a <-
ggplot(pdp_Yield_CT, aes(x=Yield_CT, y=yhat, color=algorithm)) +
  geom_line(size=1) +
  scale_color_manual(values=c("Linear model"="darkgrey",
                              "Decision Tree"="orange",
                              "Random Forests"="darkgreen",
                              "Gradient Boosting"="darkblue"))  +
  ylab("Partial dependence") +
  theme_bw() +
  theme(legend.position=c(0.7,0.7),
        axis.title.x = element_blank(),
        axis.text.x  = element_blank())
  

Fig05b <-
ggplot(pdp_Tmax, aes(x=Tmax, y=yhat, color=algorithm)) +
  geom_line(size=1) +
  scale_color_manual(values=c("Linear model"="darkgrey",
                              "Decision Tree"="orange",
                              "Random Forests"="darkgreen",
                              "Gradient Boosting"="darkblue"))  +
  ylab("Partial dependence") +
  theme_bw() +
  theme(legend.position="none", 
                                       axis.title.x = element_blank(),
                                       axis.text.x  = element_blank())
        

Fig05c <- ggplot(data_train, aes(x=Yield_CT)) + geom_histogram() + theme_bw()
Fig05d <- ggplot(data_train, aes(x=Tmax)) + geom_histogram() + theme_bw()

Fig05 <- Fig05a + Fig05b + 
         Fig05c + Fig05d +
  plot_annotation(tag_levels = "a") +
  plot_layout(heights=c(9,1))

Fig05



# Yield_CT * Tmax
pdp_Yield_CT2   <- rbind(
  model.lm %>%  partial(pred.var=c("Yield_CT","Tmax")) %>% cbind(., algorithm = "Linear model"),
  model.cart %>%  partial(pred.var=c("Yield_CT","Tmax"), approx=T) %>% cbind(., algorithm = "Decision Tree"),
  model.rf %>%  partial(pred.var=c("Yield_CT","Tmax"), approx=T) %>% cbind(., algorithm = "Random Forests"),
  model.gbm %>%  partial(pred.var=c("Yield_CT","Tmax"), approx=T, n.trees=500)  %>% cbind(., algorithm = "Gradient Boosting")
) 

pdp_Yield_CT2$algorithm <- factor(pdp_Yield_CT2$algorithm, levels=c("Linear model","Decision Tree","Random Forests","Gradient Boosting"))
pdp_Yield_CT2 <- 
  pdp_Yield_CT2  %>% 
  mutate(Tmax_class = case_when(
    Tmax > 32 ~ "Tmax_high",
    Tmax < 32 ~ "Tmax_low"
  ))


Fig06 <-
ggplot(pdp_Yield_CT2, aes(x=Yield_CT, y=yhat, 
                          color=algorithm,
                          linetype=Tmax_class)) +
  facet_wrap(vars(algorithm)) +
  geom_line(stat="summary", fun=mean, size=1) +
  ylab("Partial dependence") +
  theme_bw()  +
scale_color_manual(values=c("Linear model"="darkgrey",
                            "Decision Tree"="orange",
                            "Random Forests"="darkgreen",
                            "Gradient Boosting"="darkblue"))  

Fig06
  
# 2-way interactions
pdp_lm <- model.lm %>%  partial(pred.var=c("Yield_CT","Tmax")) %>% autoplot + labs(title="Linear model")
pdp_cart <- model.cart %>%  partial(pred.var=c("Yield_CT","Tmax"), approx=T) %>% autoplot + labs(title="Decision Tree")
pdp_rf <- model.rf %>%  partial(pred.var=c("Yield_CT","Tmax"), approx=T) %>% autoplot + labs(title="Random Forests")
pdp_gbm <- model.gbm %>%  partial(pred.var=c("Yield_CT","Tmax"), approx=T, n.trees=500) %>% autoplot + labs(title="Gradient Boosting")

Fig07 <-
  pdp_lm + pdp_cart + pdp_rf + pdp_gbm

Fig07



#----------------------------
# LIME
# resampling
data_lime <- data_train %>% select(-Yield_change)
set.seed(7); sample_id <- sample(c(1:nrow(data_lime)),size=1)
sample_data <- data_lime[sample_id,]
gower_power <- 2
dist_fun <- "canberra"
kernel_width <- 2
k <- 5

predictor.lm <- Predictor$new(model.lm, data = data_lime, y = data_train$Yield_change)
lime.explain.lm <- LocalModel$new(predictor.lm, k=k, x.interest = sample_data, dist.fun = dist_fun, gower.power = gower_power, kernel.width = kernel_width)

predictor.cart <- Predictor$new(model.cart, data = data_lime, y = data_train$Yield_change)
lime.explain.cart <- LocalModel$new(predictor.cart, k=k, x.interest = sample_data, dist.fun = dist_fun, gower.power = gower_power, kernel.width = kernel_width)

predictor.rf <- Predictor$new(model.rf, data = data_lime , y = data_train$Yield_change)
lime.explain.rf <- LocalModel$new(predictor.rf, k=k, x.interest = sample_data, dist.fun = dist_fun, gower.power = gower_power, kernel.width = kernel_width)

predictor.gbm <- Predictor$new(model.gbm, data = data_lime, y = data_train$Yield_change)
lime.explain.gbm <- LocalModel$new(predictor.gbm, k=k, x.interest = sample_data, dist.fun = dist_fun, gower.power = gower_power, kernel.width = kernel_width)

lime_point <- world_map +
  geom_point(data=data_train[sample_id,], aes(x=Longitude, y=Latitude), 
             shape=23, size=3, color="red", fill="red") 
  

layout <- "
ABC
DEE
"

Fig08 <-
plot(lime.explain.lm) + plot(lime.explain.cart) +
  plot(lime.explain.rf) + plot(lime.explain.gbm) + 
  lime_point +
  plot_layout(design = layout, heights=c(1,1)) +
  plot_annotation(tag_levels = 
                    list(c("a Linear Model", "b Decision Tree",
                           "c Random Forests", "d Gradient Boosting",
                           "e Site Location (Actual value = -0.239)")))&     
  theme_bw() &
  theme(title =element_text(size=10),
        plot.tag.position  =  c(0.3, 1.05), 
        plot.margin = unit(c(.3,.2,0.2,.1), "cm")) 

Fig08
