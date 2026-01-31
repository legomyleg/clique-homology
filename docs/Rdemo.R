#uncomment and run this line to install the package
#install.packages("tidyverse") 
#install.packages("igraph")
library(tidyverse) #import the package

#initialize the data frame
df <- data.frame("x" = seq(1, 10), "y" = seq(1, 10))

#scatterplot
ggplot(df, aes(x = x, y = y)) + 
geom_point(size = 2, color = "magenta") + # plot the scatter plot
labs(
    xlab = "X",
    ylab = "Y",
    title = "Y = X"
) + 
theme_minimal() + # try the different themes
# fit a linear model
geom_smooth(method = "lm", se = FALSE, color = "maroon") 
