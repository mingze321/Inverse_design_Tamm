# Import the package
# Important note: This code is developed by Mingze He at Vanderbilt University, USA, under supervision of Prof. Joshua D. Caldwell.
# Important note: The algorithm uses Tensorflow to back-propagate to inversely design the parameters of THIN FILM OPTICS.
# Important note: Here is an example to design a structure normally called "Tamm plasmon", which is a multi-dielectric layer stacks on a metallic layer.
# If you use the code, please cite our manuscript
# If you have any questions, please contact us via   josh.caldwell@vanderbilt.edu    mingze.he@vanderbilt.edu


# Important note: Tensorflow used here is TF. 2.3.0, CPU mode is used

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import  tensorflow as tf
import math
import numpy as np
import matplotlib.pyplot as plt
print (tf.__version__)
from tensorflow.keras import Model
import tensorflow_probability as tfp
from scipy.stats import norm
from scipy.stats import cauchy
from scipy.interpolate import interp1d
# import pylab as pl

# Set CPU as available physical device
my_devices = tf.config.experimental.list_physical_devices(device_type='CPU')
tf.config.experimental.set_visible_devices(devices= my_devices, device_type='CPU')

# To find out which devices your operations and tensors are assigned to
tf.debugging.set_log_device_placement(True)

# Important Note: Because the algorithm is based on back-propagation, sometimes it might be stuck at LOCAL MINIMUM. Running the code for several times might boost the performance.
# Important NOTE: unless specified, all the units are: thickness in nm, frequency of light in nm, refractive index of material in n and k (complex n)
# The ploting might be converted into wavenumber, but all the calculation are in nm (wavelength)
# Important Node: the parameters of strucuture is printed instead of exported into a file

# Important Note: TE mode is calculated.
# Important Note: hyper-parameters: learning rate, step, learning epoches, the components of loss function, batch size


# Calculate the characteristic matrix of a single layer
def Ch_Matrix(theta_in, n0, n1, d1,n2, k0):
    one_m = tf.constant(1.0, tf.float64)
    zeros_m = tf.constant(0.0, tf.float64)
    imag = tf.complex(zeros_m, one_m)

    W,L = n0.shape

    zeros_temp=tf.constant (0.0, dtype=tf.float64, shape= (W,1 ))
    zeros_complex= tf. complex (zeros_temp, zeros_m)

    ones_temp = tf.constant(1.0, dtype=tf.float64, shape= (W,1 ))
    ones_complex = tf.complex(ones_temp, zeros_m)

    k0 = tf.complex(k0, zeros_m)
    di = tf.complex(d1, zeros_m)
    di=tf.reshape(di, (W,1))
    n1 = tf.reshape(n1, (W, 1))
    n2 = tf.reshape(n2, (W, 1))
    cos1=1/n1*tf.math.sqrt (n1**2- (n0 * tf.math.sin (theta_in))**2)
    cos2=1/n2*tf.math.sqrt (n2**2- (n0 * tf.math.sin (theta_in))**2)
    # print (cos1, 'cos1')
    rs12= (n1*cos1-n2*cos2)/ (n1*cos1+n2* cos2)

    ts12= 2*n1*cos1/ (n1*cos1+n2* cos2)
    ts12 = tf.expand_dims(ts12, axis=2)

    optical_pass= di*k0*n1*cos1
    # print (optical_pass.shape, di.shape, k0.shape, n1.shape, cos1.shape, W, 'all shapes')
    Matrix= [[ tf.math.exp (-imag * optical_pass), zeros_complex], [zeros_complex,tf.math.exp (imag * optical_pass) ]]
    Matrix = tf.reshape(Matrix, shape=(4, W))
    Matrix = tf.reshape(Matrix, shape=(2,2, W))
    Matrix = tf.transpose(Matrix, [2, 1, 0])

    M_t2=[[ones_complex, rs12], [rs12, ones_complex] ]
    M_t2 = tf.reshape(M_t2, shape=(4, W))
    M_t2 = tf.reshape(M_t2, shape=(2,2, W))
    M_t2 = tf.transpose(M_t2, [2, 1, 0])

    M2=tf.linalg.matmul (Matrix, M_t2)/ ts12

    return M2

# Transfer matrix calculation of all the layers
def TMM(theta_in,n0,n,d,k0):

    temp=n
    W,L=temp.shape
    L=L-1

    one_m=tf.constant(1.0,tf.float64, shape=[W,1] )
    zeros_m=tf.constant(0.0,tf.float64, shape=[W,1])
    one_m_complex=tf.complex(one_m, zeros_m)

    cos0=tf.math.cos (theta_in)
    cos1=1/n[0,0]* tf.math.sqrt (n[0,0]**2- (n0* tf.math.sin (theta_in))**2)
    ts01=2*n0*cos0/ (n0*cos0 +n[0,0] *cos1)
    ts01=tf.expand_dims(ts01, axis=2)
    rs01= (n0*cos0 -n[0,0] *cos1)/(n0*cos0 +n[0,0] *cos1)
    # Take care of the matrix shape conversion
    M01= [[one_m_complex, rs01 ], [rs01 , one_m_complex]]
    M01 = tf.reshape(M01, shape=(4, W))
    M01 = tf.reshape(M01, shape=(2,2, W))
    M01 = tf.transpose(M01, [2, 1, 0])


    M0=M01 /ts01
    Transfer=M0

    # Loop the layers
    for layer_ind in range (L):
        Temp_transfer_matrix = Ch_Matrix(theta_in,n0,n[:,layer_ind],d[:,layer_ind],n[:,layer_ind+1],k0)
        # print ('transfer matrix',Temp_transfer_matrix)
        # print ('medium transfer', layer_ind, Temp_transfer_matrix)
        Transfer=tf.linalg.matmul(Transfer,Temp_transfer_matrix)

    return Transfer


# Calculate the reflectance from the TMM
def reflectance(wavelength, theta_in, n0, n, d):
    one_m = tf.constant(1.0, tf.float64)
    zeros_m = tf.constant(0.0, tf.float64)

    theta_in = tf.complex(theta_in, zeros_m)

    k0 = 1 / wavelength * 2 * 3.1415926
    k0=tf.expand_dims(k0, axis=1)

    Matrix_overall = TMM(theta_in, n0, n, d, k0)

    m00=Matrix_overall[:, 0, 0]
    m10=Matrix_overall[:, 1, 0]
    refte= m10/m00

    Reflectance =(tf.math.abs (refte))**2
    # Important note: the reflectance calculated here is <=1

    return Reflectance


# Define Drude Model
# carrier density is carrier density of CdO
# Model of CdO is taken from Nolen, J. Ryan, et al. "Ultraviolet to far-infrared dielectric function of n-doped cadmium oxide thin films." Physical Review Materials 4.2 (2020): 025202.
# Frequency is in cm-1
def Drude(carrier, wavelength):
    k=10000000/wavelength #k is in wavenumber
    w = (2.998e+10) * k  # Frequency - Hz
    #     l=len(frequency)
    # Constants
    q = 1.60217662e-19
    m = 9.10938e-31
    eps_o = 8.854e-12
    C_CdO = 1.47
    mo_CdO = 0.1
    #     CdO Properties
    n1 = carrier* (1e+20);  # CC in cm^-3
    print (n1, 'carrier_concnetration')
    mu1 = 200.0 # mobility in cm^2/(V-s)
    eps_CdO_inf = 5.1  # High frequency permittivity

    n1_eff = n1 * (1e-20)
    m_eff_1 = mo_CdO * (1 + (2 * C_CdO) * ((0.19732697 ** 2) / (mo_CdO * 510998.5)) * (3 * 3.14 * n1_eff * 10 ** 8) ** (
                2 / 3)) ** (1 / 2)
    wp1 = (1000 / (2 * 3.14)) * ((n1 * q ** 2) / (m_eff_1 * m * eps_o)) ** 0.5;  # plasma frequency of layer 1
    gamma1 = (10000 / (2 * 3.14)) * (q / (mu1 * m_eff_1 * m))  # damping of layer 1 in Hz
    # print (gamma1, w, wp1, 'gamma1, w, wp1')
    eps_1_real = eps_CdO_inf - (wp1 ** 2 / (w ** 2 + gamma1 ** 2))
    eps_1_imag = (gamma1 * wp1 ** 2) / (w * (w ** 2 + gamma1 ** 2))
    DF_1 = tf.complex(eps_1_real, eps_1_imag)
    refractive_index=tf.math.sqrt (DF_1)

    return refractive_index



def set_target():
    # The following set target function is to set a gaussian shaped target
    # Note: the shape of Tamm plasmon is normally Lorentz shaped, so please use "Cauchy.pdf" from scipy.stats for best performance

    # Define the frequency range and the target spectra
    wavenumber_unshuffled = np.arange(1500, 3500, 1)
    wavelength_unshuffled = 10000000 / wavenumber_unshuffled
    frequency_length = len(wavelength_unshuffled)
    wavelength_unshuffled = wavelength_unshuffled.reshape(frequency_length, )
    wavelength_unshuffled = np.float64(wavelength_unshuffled)
    wavelength_plot = wavelength_unshuffled.astype(np.float64)

    # Set target reflectance function
    Ref_target_uns = np.ones((frequency_length, 1))*100

    # The following is to set several dips.

    absorption2 = cauchy.pdf(wavelength_unshuffled, 4237.28814,30)
    absorption2 = absorption2 / (absorption2.max()) * 100
    absorption2 = np.reshape(absorption2, (frequency_length, 1))

    absorption3 = norm.pdf(wavelength_unshuffled, 3500, 20)
    absorption3 = absorption3 / (absorption3.max()) * 100
    absorption3 = np.reshape(absorption3, (frequency_length, 1))

    Ref_target_uns = Ref_target_uns -absorption2-absorption3
    Ref_target_uns=Ref_target_uns.astype(np.float64)

    wavelength_plot = np.reshape(wavelength_plot, [frequency_length, 1])
    important_id = np.where(Ref_target_uns < 95)
    wave_important = wavelength_plot[important_id]
    Ref_important = Ref_target_uns[important_id]

    print(len(wave_important), 'data point in resonance')
    plt.figure()
    plt.plot(10000000/wavelength_plot, Ref_target_uns)
    plt.xlabel('frequency cm-1')
    plt.ylabel('Reflectivity (%)')

    plt.title('target spectra')
    plt.show()


    return wavelength_plot, Ref_target_uns,wave_important, Ref_important


# Important note: the optical structure is defined here. The layer_length is the total layer number without the substrate
# Important note: layer thickness of each layer is limited to 50-850 nm, by a sigmoid function, and the purpose is to make the design realistic for growing.
class MyModel(Model):
# class MyModel (Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # Important note: Randomly initialize the parameters
        # Note, all the values are in float64 and complex128
        self.layer_length=11
        self.a=np.random.random( size=self.layer_length )-0.5
        self.A= tf.Variable(self.a,trainable=True,dtype= tf.float64)
        self.Carrier=np.random.random()-0.5
        self.Carrier= tf.Variable(self.Carrier,trainable=True, dtype= tf.float64)
        self.drude=Drude



    def call(self, x):
        frequency_length=tf.size(x)
        x=tf.reshape(x, [frequency_length,1])
        one_m = tf.constant(1.0, tf.float64)
        zeros_m = tf.constant(0.0, tf.float64)
        x = tf.dtypes.cast(x, tf.float64)
        carrier1 = self.Carrier
        # Sigmoid function is used to limit the range of carrier concentration and the layer thickness

        carrier1 = tf.keras.activations.sigmoid(carrier1) * 3.6 + 0.4

        Ge =tf.complex(4.*one_m,zeros_m)
        SiO = tf.complex(2.25*one_m,zeros_m)
        vac=tf.complex(1.*one_m,zeros_m)

        CdO1=self.drude( carrier=carrier1, wavelength=x)
        sub=SiO
        n0 = tf.complex(one_m, zeros_m)
        theta_in = one_m * 0 / 180 * 3.1415926

        print (Ge, 'Ge dielectric function')
        d = self.A[-self.layer_length:]

        d = tf.keras.activations.sigmoid(d) * 8+0.5
        d=d*100.

        print(d, 'thickness information')
        print (carrier1, 'carrier concentration')

        # Define the substrate, indcident angle

        # Important note: n1 defines the matrial order. For example, here the structure is: Ge-SiO-Ge...-Ge-CdO (with back-calculated carrier concentration)-substrate (highly doped CdO)
        # First layer is vac to make the calculation stable
        n1= [vac, Ge, SiO, Ge, SiO, Ge,SiO, Ge,SiO, Ge]


        n0 = tf.broadcast_to(n0, [frequency_length, 1])
        n1 = tf.broadcast_to(n1, [frequency_length, self.layer_length-1])
        ns = tf.broadcast_to(sub, [frequency_length, 1])
        d = tf.broadcast_to(d, [frequency_length, self.layer_length])
        x = tf.reshape(x, [frequency_length, ])
        # Concat the dielectric and CdO: (N1 and CdO1 and substrate ns)
        n = tf.concat([n1, CdO1, ns], axis=1)

        ref = reflectance(x, theta_in, n0, n, d)

        return ref*100


def Linf (ypred, y):
    res=(ypred- y)**2
    res=tf.reduce_max(res)
    return res



#set the target spectra
wavelength,ref_target, wavelength_important, ref_important= set_target()

# Dataloader, load data into TF data-format
train_ds=tf.data.Dataset.from_tensor_slices((wavelength,ref_target)).shuffle(5000).batch(500)
train_ds2=tf.data.Dataset.from_tensor_slices(( wavelength_important, ref_important)).shuffle(5000).batch(512)

# Define the model
model=MyModel()
tf.keras.backend.set_floatx('float64')
# Define the loss functions. The loss functions are a mixure of meansquareerror + meanabsolute+ Linf
loss_object = tf.keras.losses.MeanSquaredError()
loss_object1=tf.keras.losses.MeanAbsoluteError()
loss_object3=Linf

#Important Note: The back propagation is separated into two parts: "important" and "all", the important is the resonance frequency, and all the full frequency range
initial_learning_rate =0.05
# The learning rate is exponentially decaying, increase the decay steps and epochs for the best performance
lr_schedule =tf. keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=100,
    decay_rate=0.7,
    staircase=True)

lr_schedule_small_region =tf. keras.optimizers.schedules.ExponentialDecay(
    0.01,
    decay_steps=100,
    decay_rate=0.7,
    staircase=True)

# important Adam optimizer is used.
optimizer1=tf.keras.optimizers.Adam(learning_rate=lr_schedule_small_region)
optimizer2=tf.keras.optimizers.Adam(learning_rate=lr_schedule)

train_loss = tf.keras.metrics.Mean(name='train_loss')
test_loss = tf.keras.metrics.Mean(name='test_loss')

@tf.function
def train_step1(images, labels):
  with tf.GradientTape() as tape:
    # training=True is only needed if there are layers with different

    print (model.trainable_variables, 'all variables')
    predictions = model(images, training=True)
    # Important note: here define the loss function component.
    loss = loss_object(labels, predictions)+loss_object3(labels, predictions)*0.01
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer1.apply_gradients(zip(gradients, model.trainable_variables))

  train_loss(loss)

@tf.function
def train_step2(images, labels):
  with tf.GradientTape() as tape:

    print (model.trainable_variables, 'all variables')
    predictions = model(images, training=True)
    loss = loss_object(labels, predictions)+loss_object3(labels, predictions)*0.01
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer2.apply_gradients(zip(gradients, model.trainable_variables))

  train_loss(loss)


EPOCHS1 = 200
EPOCHS = 450

# Backpropagation process: firstly backpropagate the resonance part, then the full frequency range
for epoch in range(EPOCHS1):
  # Reset the metrics at the start of the next epoch
  train_loss.reset_states()
  test_loss.reset_states()

  for images, labels in train_ds2:
      train_step1(images, labels)

  if epoch%10==0 or epoch==0:
      template = 'Epoch {}, Loss: {}'
      print(template.format(epoch + 1,
                            train_loss.result(),

                            ))

for epoch in range(EPOCHS):
  # Reset the metrics at the start of the next epoch
  train_loss.reset_states()
  test_loss.reset_states()

  for images, labels in train_ds:
    train_step2(images, labels)




  if epoch%10==0 or epoch==0:
      template = 'Epoch {}, Loss: {}'
      print(template.format(epoch + 1,
                            train_loss.result(),

                            ))



# Plot the designed spectra

frequency_len= len (wavelength)
designed_spectra= np.zeros (frequency_len,)

designed_spectra= model (wavelength)



plt.figure()
plt.plot(10000000/wavelength, designed_spectra, label='Spectra from inverse design')
plt.plot (10000000/wavelength, ref_target, label='target spectra')
plt.title('designed')
plt.xlabel('frequency cm-1')
plt.ylabel('Reflectivity (%)')
plt.legend()

plt.show()


# Important Save the designed spectra
Target_output=np.zeros((frequency_len,2))
Target_output[:,0]=wavelength[:,0]
Target_output[:,1]=ref_target[:,0]

Designed_output=np.zeros((frequency_len,2))
Designed_output[:,0]=wavelength[:,0]
Designed_output[:,1]=designed_spectra

# Print out the parameters of the optical structure, layer thickness and carrier concentration
print (model.trainable_variables)

np.savetxt('target_spectra.csv',Target_output, delimiter=",")
np.savetxt('designed_spectra.csv',Designed_output, delimiter=",")
