import tensorflow as tf

def weibull_loglikelihood_continuous(a_, b_, y_, u_,name=None):
    ya = tf.math.divide(y_+1e-35,a_)
    return(
        tf.multiply(u_,
               tf.math.log(b_)+tf.multiply(b_,tf.math.log(ya))
              )- 
        tf.math.pow(ya,b_))

def weibull_loglikelihood_discrete(a_, b_, y_, u_, name=None):
    with tf.name_scope(name):
        hazard0 = tf.math.pow(tf.math.divide(y_+1e-35,a_),b_) 
        hazard1 = tf.math.pow(tf.math.divide(y_+1,a_),b_)
    return(tf.multiply(u_,tf.math.log(tf.exp(hazard1-hazard0)-1.0))-hazard1)

def weibull_beta_penalty(b_,location = 10.0, growth=20.0, name=None):
    # Regularization term to keep beta below location
    with tf.name_scope(name):
        scale = growth/location
        penalty_ = tf.exp(scale*(b_-location))
    return(penalty_)