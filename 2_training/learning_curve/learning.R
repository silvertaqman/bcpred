#!/usr/bin/env Rscript
library(readr)
library(dplyr)
library(tidyr)
library(ggplot2)

# Learning curve
# read_csv("../predictions.csv") %>%
ler <- read_csv("learning_curve.csv") %>%
	select(train_size, models, starts_with("train_scores"), starts_with("test_scores")) %>%
	rowwise() %>%
	transmute(
		train_size = train_size,
		models = models,
		train_score = mean(c_across(starts_with("train_scores"))),
		train_deviation = sd(c_across(starts_with("train_scores"))),
		test_score = mean(c_across(starts_with("test_scores"))),
		test_deviation = sd(c_across(starts_with("test_scores")))) %>%
	ggplot(aes(x=train_size))+
		  geom_ribbon(
		  	aes(
		  		ymin = train_score-train_deviation,
		  		ymax = train_score+train_deviation),
		  	fill = "grey90")+
  geom_line(aes(y = train_score, colour=models))+
		  geom_ribbon(
		  	aes(
		  		ymin = test_score-test_deviation,
		  		ymax = test_score+test_deviation),
		  	fill = "grey90")+
  geom_line(aes(y = test_score, colour=models))+
  facet_wrap(~models)+
  scale_colour_manual(
    "Modelo",
     values = c("#5E1E5B","#F89B0F","#F26F7E"))+
  theme(
    # Use gray text for the region names
    axis.text.x = element_text(color = "gray12", size = 12),
    axis.text.y = element_text(color = "gray12", size = 12),
    # Move the legend to the bottom
    legend.position = "bottom",
    # Set default color and font family for the text
    text = element_text(color = "gray12"),
    # Make the background white and remove extra grid lines
    panel.background = element_rect(fill = "white", color = "white"),
    panel.grid = element_blank(),
    panel.grid.major.x = element_blank()
  )+
  annotate("label", x=200, y=1, label="Train")+
  annotate("label", x=200, y=0.87, label="Test")

ggsave("learning_curve2.pdf",
				ler, 
				dpi=300,
				width = 4000, 
				height = 2000, 
				units = "px",
				useDingbats=FALSE)
				
# Scalability of the model

sca <- read_csv("learning_curve.csv") %>%
	select(train_size, models, starts_with("fit_times_fold")) %>%
	rowwise() %>%
	transmute(
		train_size = train_size,
		models = models,
		fit_times = mean(c_across(starts_with("fit_times_fold"))),
		deviation = sd(c_across(starts_with("fit_times_fold")))) %>%
	ggplot(aes(x=train_size, group=models))+
		  geom_ribbon(
		  	aes(ymin = fit_times-deviation, ymax = fit_times+deviation),
		  	fill = "grey90")+
  geom_line(aes(y = fit_times, colour=models))+
#  facet_wrap(~models, scales="free")+
  scale_colour_manual(
    "Modelo",
     values = c("#5E1E5B","#F89B0F","#F26F7E"))+
  theme(
    # Use gray text for the region names
    axis.text.x = element_text(color = "gray12", size = 12),
    axis.text.y = element_text(color = "gray12", size = 12),
    # Move the legend to the bottom
    legend.position = "bottom",
    # Set default color and font family for the text
    text = element_text(color = "gray12"),
    # Make the background white and remove extra grid lines
    panel.background = element_rect(fill = "white", color = "white"),
    panel.grid = element_blank(),
    panel.grid.major.x = element_blank()
  )
ggsave("scalability.pdf",
				sca, 
				dpi=300,
				width = 2000, 
				height = 2000, 
				units = "px",
				useDingbats=FALSE)

# Performance of the model

perf <- read_csv("learning_curve.csv") %>%
	select(models, starts_with("fit_times_fold"), , starts_with("test_score")) %>%
		pivot_longer(cols=starts_with("test_score"), names_to="test_fold", values_to="test_score") %>%
		rowwise() %>%
		transmute(
			models = models,
			test_score = test_score,
			fit_times = mean(c_across(starts_with("fit_times_fold"))),
			deviation = sd(c_across(starts_with("fit_times_fold")))) %>%
		ggplot(aes(x=fit_times, y=test_score, colour=models))+
		  geom_ribbon(
		  	aes(
		  		ymin = fit_times-deviation,
		  		ymax = fit_times+deviation),
		  	colour = "grey90")+
		  geom_point()+
			geom_line()+
		facet_wrap(~models, scales="free")+
#  facet_wrap(~models, scales="free")+
  scale_colour_manual(
    "Modelo",
     values = c("#5E1E5B","#F89B0F","#F26F7E"))+
  theme(
    # Use gray text for the region names
    axis.text.x = element_text(color = "gray12", size = 12),
    axis.text.y = element_text(color = "gray12", size = 12),
    # Move the legend to the bottom
    legend.position = "bottom",
    # Set default color and font family for the text
    text = element_text(color = "gray12"),
    # Make the background white and remove extra grid lines
    panel.background = element_rect(fill = "white", color = "white"),
    panel.grid = element_blank(),
    panel.grid.major.x = element_blank()
  )

ggsave("scalability.pdf",
				sca, 
				dpi=300,
				width = 2000, 
				height = 2000, 
				units = "px",
				useDingbats=FALSE)

