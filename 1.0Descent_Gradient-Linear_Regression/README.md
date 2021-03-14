<h1 align="center">LINEAR REGRESSION - BASIC CONCEPTS</h1>
<div align="center">
  <p>
    <strong>Descent Gradient
  </strong>
  </p>
</div>

# Linear Regression - Descent Gradient

## Understant Descent Gradient by a mathematic perspective

### Descent Gradient - Minimizing the Cost Function

We have a Cost Function that computes the Linear Regression Model error rate, where one of the main goals during the training is to minimize the value of this Cost Function.

In other words, the Cost Function is calculated by the difference between the model prediction value (one of the hypothesis) and the expected value, the result is squared to remove any negative value, and then, we sum all the squared differences, because we have many data points for each interaction. With the Cost Function value, we can adjust the weights or coefficients for the next iteration, to approximate de hypothesis h(x) to y.

But, how can we know what is the best value for the weights, for the coefficients? In this case, we use de Descent Gradient.

 <img src="https://github.com/meloandrew/Python-Data_Analytics_Machine_Learning/blob/master/1.0Descent_Gradient-Linear_Regression/images/gradiente2.png"
      alt = "Gradient"
      style="width:600px;height:300px;"/>

On the image above, J represents the Cost Function with the two parameters learned during the training, which are used to compare h(x) to y. We have two black lines, that Descent from the red region to the blue region, these lines represent the Descent Gradient.

For each interaction, we use the Cost Function value to update the values of the coefficients, but we use the concept of the directional partial derivative to know if we are following the right direction or not, in other words, to know the best difference to update the weights for the next interaction. If we did not have the Descent Gradient, the Cost Function would update its parameters, but without any direction, or metrics to know if the updated coefficients would minimize the error or not.

The weight is not updated by the Cost Function. We use the Cost Function to feed another mathematic step that will update the weights for the next iteration, to minimize the J value.

The Descent Gradient is great when the parameters can not be analytically obtained (using linear algebra, for example) and should be researched by an optimizing algorithm.

### Descent Gradient - Local Maximum and Minimum

The mathematic Functions may have "hills and valleys": regions where they assume maximum and minimum values. For example, back at school, while studying free-fall motion problems, the object movement Function assumes a parabole form (in major problems), where its maximum points represent the moment where the variation between space and time assumes zero value or the kinetic energy is equal to the gravitation potential energy (not considering other involved elements in this case), or the object movement velocity is equal to zero.

 <img src="hhttps://github.com/meloandrew/Python-Data_Analytics_Machine_Learning/blob/master/1.0Descent_Gradient-Linear_Regression/images/func1.png"
      alt = "Descent_Gradient_func1"
      style="width:600px;height:300px;"/>

The concept of global or absolute maximum and minimum states there is only one maximum and minimum point in the Function, but it can have more than one local maximum or local minimum.

<img src="https://github.com/meloandrew/Python-Data_Analytics_Machine_Learning/blob/master/1.0Descent_Gradient-Linear_Regression/images/func2.png"
      alt = "Maximum_and_Minimum_local"
      style="width:600px;height:300px;"/>

The image bellow represents the concept of Descent Gradient:

<img src="https://github.com/meloandrew/Python-Data_Analytics_Machine_Learning/blob/master/1.0Descent_Gradient-Linear_Regression/images/gradiente1.png"
      alt = "Gradient_1"
      style="width:600px;height:300px;"/>

The blue line represents the Cost Function, the one we want to minimize. We need to find the weight values, w, that minimize this Function, and thus, leading to a lower error rate. In order to do that, the global minimum point of the Cost Function must be found. We usually start with a random weight to get its best value for each iteration to minimize the Cost Function.

But, we have a little problem.

During this process, the Descent Gradient can find the local minimum and assumes this is the best value for the Cost Function. To avoid this, we use a learning rate to train the linear regression model, which represents the size of each iteration (the steps) until it reaches the global minimum. The weight distance must be defined to train the model.

## Python Experiment

We have 3 essential elements in the learning processing:

* Pattern existence
* Data
* No Cost Function defined

For didactic purposes, we will violate one of these rules. We will bring you a mathematic Function that already has a solution.

The Function is y = (x + 5)², and we want to find its minimum point. We can do that without using Python or any programming language.

First, we find its derivative, which is 2(x+5) = 2x + 10. If the derivative is equal to zero, the minimum point for this Function is -5.

Let's find this result using the Descent Gradient with Python Language.

We will define the initial weight value that the Descent Gradient will use:

```python
cur_x = 3
```

The learning rate is the distance between each step given by the Descent Gradient to update the weight parameter until it reaches the Cost Function global minimum. Our learning rate will be equal to 0.01:

```python
rate = 0.01
```

We will define a rule to stop the algorithm, in other words, the Descent algorithm will stop its iterations when the difference between the previous and the actual value of the weight is smaller than 0.000001, or when the number of iterations is greater than 10 thousand. The first parameter is called precision and the second limits the number of iterations:

```python
precision = 0.000001
previous_step_size = 1 
max_iters = 10000
```

The next step is to define the iterations counter and the Function Gradient (first derivative of our Function or its partial derivative):

```python
iters = 0
gf: lamba x: 2*(x+5)
```

With the parameters already defined, we have everything to train our algorithm. For that, we will use a While loop controled by the number of iterations and the precision previously defined:

```python
while previous_step_size > precision and iters < max)iters:

  # Store x current value (weight):
  prev_x = cur_x

  # Applying the Descent Gradient
  #:
  cur_x = cur_x - rate * gf(prev_x)

  # Increment iteration number:
  iters = iters + 1

  # Print iterations number and respective weight:
  print("Iteration" iters, "\nWeight Value is equal to ", cur_x)

print("\nFunction Minimum Value: ", cur_x)
```

<img src="https://github.com/meloandrew/Python-Data_Analytics_Machine_Learning/blob/master/1.0Descent_Gradient-Linear_Regression/images/gradienteoutput.png"
      alt = "Gradient_Output"
      style="width:600px;height:300px;"/>




