#![allow(dead_code)]
extern crate nalgebra as na;

use na::base::{DMatrix, DVector};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
pub struct Network<T> {
    pub node_count: Vec<usize>,
    pub input_data: Vec<T>,
    pub alpha: f64,             // learning rate
    pub gamma: f64,             // patience factor
    weights: Vec<DMatrix<f64>>, // row only (col vector)
    biases: Vec<DVector<f64>>,  // row,col
    layers: Vec<DVector<f64>>,  // row,col //stores z_j as in a_j = phi(z_j)
    in_ty: PhiT,
    out_ty: PhiT, // to select internal activation function used
}
#[derive(Serialize, Deserialize, Clone)]
pub struct Gradient {
    db: Vec<DVector<f64>>,
    dw: Vec<DMatrix<f64>>,
}

pub type Grad = Gradient;

impl Gradient {
    pub fn new() -> Self {
        Gradient {
            dw: Vec::new(),
            db: Vec::new(),
        }
    }
}

fn phi(ty: &PhiT) -> fn(f64) -> f64 {
    match ty {
        PhiT::Sigmoid => sigmoid,
        PhiT::ReLU => relu,
        PhiT::ReLU6 => relu6,
        PhiT::LReLU => lrelu,
        PhiT::Tanh => tanh,
        PhiT::SoftPlus => softplus,
        PhiT::FSigmoid => fsigmoid,
        PhiT::Linear => linear,
        //PhiT::SPReLU => sprelu,
    }
}

fn dphi(ty: &PhiT) -> fn(f64) -> f64 {
    match ty {
        PhiT::Sigmoid => dsigmoid,
        PhiT::ReLU => drelu,
        PhiT::ReLU6 => drelu6,
        PhiT::LReLU => dlrelu,
        PhiT::Tanh => dtanh,
        PhiT::SoftPlus => dsoftplus,
        PhiT::FSigmoid => dfsigmoid,
        PhiT::Linear => dlinear,
        //PhiT::SPReLU => dsprelu,
    }
}

#[derive(Serialize, Deserialize)]
enum PhiT {
    Sigmoid,
    ReLU,
    ReLU6,
    LReLU,
    Tanh,
    SoftPlus,
    FSigmoid,
    Linear,
    //PReLU,
    //SPReLU,
}

/*
pub fn sprelu(x: f64) -> f64 {
    x / (1.0 + x.abs())
}

pub fn dsprelu(a: f64, x: f64) -> f64 {
    //not actually its derivative
    fsigmoid(x) * (1.0 - fsigmoid(x))
}
pub fn prelu(a: f64, x: f64) -> f64 {
    x / (1.0 + x.abs())
}

pub fn dprelu(a: f64, x: f64) -> f64 {
    //not actually its derivative
    fsigmoid(x) * (1.0 - fsigmoid(x))
}
*/

pub fn linear(x: f64) -> f64 {
    x
}

pub fn dlinear(_x: f64) -> f64 {
    1.0
}

pub fn relu6(x: f64) -> f64 {
    if x < 0.0 {
        0.000000001 * x
    } else if x < 6.0 {
        x
    } else {
        6.0
    }
}

pub fn drelu6(x: f64) -> f64 {
    if x < 0.0 {
        0.000000001
    } else if x < 6.0 {
        1.0
    } else {
        0.000000001
    }
}

pub fn fsigmoid(x: f64) -> f64 {
    x / (1.0 + x.abs())
}

pub fn dfsigmoid(x: f64) -> f64 {
    //not actually its derivative
    fsigmoid(x) * (1.0 - fsigmoid(x))
}

pub fn softplus(x: f64) -> f64 {
    f64::ln_1p(f64::exp(x))
}

pub fn dsoftplus(x: f64) -> f64 {
    1.0 / (1.0 + f64::exp(-x))
}

pub fn relu(x: f64) -> f64 {
    if x >= 0.0 {
        x
    } else {
        0.0
    }
}

pub fn drelu(x: f64) -> f64 {
    if x >= 0.0 {
        1.0
    } else {
        0.0
    }
}

pub fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + f64::exp(-x))
}

pub fn dsigmoid(x: f64) -> f64 {
    sigmoid(x) * (1.0 - sigmoid(x))
}

pub fn lrelu(x: f64) -> f64 {
    if x >= 0.0 {
        x
    } else {
        x * 0.01
    }
}

pub fn dlrelu(x: f64) -> f64 {
    if x >= 0.0 {
        1.0
    } else {
        0.01
    }
}

pub fn tanh(x: f64) -> f64 {
    x.tanh()
}

pub fn dtanh(x: f64) -> f64 {
    x.tanh().mul_add(-x.tanh(), 1.0)
}

const DEFAULT_ALPHA: f64 = 0.01;
const DEFAULT_GAMMA: f64 = 0.95;
const DEFAULT_IN_PHI: PhiT = PhiT::ReLU6;
const DEFAULT_OUT_PHI: PhiT = PhiT::Linear;
// idk how to implement partial defaults, so document here
// alpha = 0.05, gamma = 0.95

impl<T: InputType> Network<T> {
    //internal node count describes internal layer
    pub fn new_default(internal_nodes: Vec<usize>, input_data: Vec<T>) -> Self {
        let mut weights: Vec<DMatrix<f64>> = Vec::with_capacity(internal_nodes.len());
        let mut biases: Vec<DVector<f64>> = Vec::with_capacity(internal_nodes.len() + 1);
        let mut layers: Vec<DVector<f64>> = Vec::with_capacity(internal_nodes.len() + 1);
        let mut j: usize = input_data.len(); //previous node count
        let mut node_count: Vec<usize> = vec![input_data.len()];
        weights.push(DMatrix::new_random(j, j));
        biases.push(DVector::new_random(j));
        layers.push(DVector::zeros(j));
        for i in internal_nodes {
            weights.push(DMatrix::new_random(i, j)); // M[i]: layer[j] -> layer[i]
            biases.push(DVector::new_random(i));
            layers.push(DVector::zeros(i));

            node_count.push(i);
            j = i;
        }
        return Network {
            node_count,
            input_data,
            alpha: DEFAULT_ALPHA,
            gamma: DEFAULT_GAMMA,
            weights,
            biases,
            layers,
            in_ty: DEFAULT_IN_PHI,
            out_ty: DEFAULT_OUT_PHI,
        };
    }

    pub fn set_input(&mut self, input_data: Vec<T>) {
        self.input_data = input_data;
    }

    pub fn get_z_output(&mut self) -> Vec<f64> {
        self.layers[self.node_count.len() - 1]
            .data
            .as_vec()
            .to_vec()
    }

    pub fn get_phiz_output(&mut self) -> Vec<f64> {
        self.layers[self.node_count.len() - 1]
            .map(phi(&self.out_ty))
            .data
            .as_vec()
            .to_vec()
    }

    pub fn get_pi_output(&mut self) -> Vec<f64> {
        softmax(
            self.layers[self.node_count.len() - 1]
                .map(phi(&self.out_ty))
                .data
                .as_vec(),
        )
    }

    pub fn forward_prop(&mut self, input_data: &Vec<T>) {
        // turn input data into usable format.
        let input_vector: DVector<f64> =
            DVector::from_iterator(input_data.len(), input_data.iter().map(|x| x.to_f64()));

        // layer l = 0
        // z_l = W_l * phi(z_{l-1}) + b_l
        self.layers[0] = &self.weights[0] * input_vector + &self.biases[0];

        // layer l = 1..L-1
        for l in 1..self.node_count.len() {
            // z_l = W_l * phi(z_{l-1}) + b_l
            self.layers[l] =
                &self.weights[l] * (self.layers[l - 1]).map(phi(&self.in_ty)) + &self.biases[l];
        }
    }

    // dCda in the function argument is more like the reward
    // pi = softmax(phi(z))
    // dpi/da = softmax'(phi(z))
    pub fn back_prop(&mut self, input: &Vec<T>, index: &usize) -> Grad {
        let mut grad = Grad::new();

        let &index = index;
        let len = self.layers[self.layers.len() - 1].len(); // output layer's node count

        let input_vector = DVector::from_iterator(input.len(), input.iter().map(|x| x.to_f64()));
        let pi_vec: Vec<f64> = self.get_pi_output(); // probability output softmax(phi(z))
        let pi_index = pi_vec[index]; // pi(a|s) corresponding to the taken action a
        let mut dpida_vec: Vec<f64> = Vec::new(); // gradient of pi_index wrt to network output a = phi(z)

        for (i, pi_i) in pi_vec.iter().enumerate().take(len) {
            if i == index {
                dpida_vec.push(pi_index * (1.0 - pi_index));
            } else {
                dpida_vec.push(-1.0 * pi_i * pi_index);
            }
        }
        let mut dpida: DVector<f64> = DVector::from_vec(dpida_vec);
        let layer_count = self.layers.len();
        for l in 1..=layer_count {
            // dpi/dz = dpi/da * da/dz
            //        = dpi/da * dphi(z)
            let z = &self.layers[layer_count - l];
            let dpidz = match l {
                1 => dpida.component_mul(&z.map(dphi(&self.out_ty))),
                _ => dpida.component_mul(&z.map(dphi(&self.in_ty))),
            };

            // dpi/db = dpi/da * da/dz * dz/db
            //        =(dpi/dz)        * 1
            grad.db.push((1.0 / pi_index) * &dpidz);

            if l != layer_count {
                // dpi/dw = dpi/da * da/dz   * dz/dw
                //        = dpi/da * phi'(z) * phi(z)
                // also note that:   (AB)^t = B^t A^t,
                //             so: (AB^t)^t = B A^t, <-- what we have here
                grad.dw.push(
                    (1.0 / pi_index)
                        * &dpidz
                        * (self.layers[layer_count - (l + 1)].map(phi(&self.in_ty))).transpose(),
                );
            } else {
                // same as above, except we use input vector
                grad.dw
                    .push((1.0 / pi_index) * &dpidz * input_vector.transpose());
            }

            // dpi/da for the next layer:
            // dpi/da'= sum dpi/da *   da/dz * dz/da'
            //        = sum  dz/da'* (dpi/da * da/dz)
            //        =        [w] * [dpi/da * phi'(z)]
            dpida = self.weights[layer_count - l].tr_mul(&dpidz);
        }
        grad.db.reverse();
        grad.dw.reverse();

        return grad;
    }

    // pushes policy interation theta = theta + alpha * dpi/dtheta (gamma was factored into the dw and db calculation)
    pub fn update(&mut self, grads: &Vec<Grad>, strides: &[i32], rewards: &[f64]) {
        let layer_count = self.layers.len();
        for i in 0..grads.len() {
            for l in 0..layer_count {
                self.biases[l] = &self.biases[l]
                    + self.alpha * f64::powi(self.gamma, strides[i]) * rewards[i] * &grads[i].db[l];
                self.weights[l] = &self.weights[l]
                    + self.alpha * f64::powi(self.gamma, strides[i]) * rewards[i] * &grads[i].dw[l];
            }
        }
    }
    pub fn update_sum(&mut self, grads: &Vec<Grad>) {
        let layer_count = self.layers.len();
        for i in 0..grads.len() {
            for l in 0..layer_count {
                self.biases[l] = &self.biases[l] + &grads[i].db[l];
                self.weights[l] = &self.weights[l] + &grads[i].dw[l];
            }
        }
    }
    pub fn sum(&mut self, grads: &Vec<Grad>, strides: &[i32], rewards: &[f64]) -> Grad {
        let mut total: Grad = grads[0].clone();
        let layer_count = self.layers.len();
        for i in 1..grads.len() {
            for l in 0..layer_count {
                total.db[l] = &total.db[l]
                    + self.alpha * f64::powi(self.gamma, strides[i]) * rewards[i] * &grads[i].db[l];
                total.dw[l] = &total.dw[l]
                    + self.alpha * f64::powi(self.gamma, strides[i]) * rewards[i] * &grads[i].dw[l];
            }
        }
        return total;
    }
}

pub trait InputType {
    fn to_f64(&self) -> f64;
}

pub fn softmax(xs: &Vec<f64>) -> Vec<f64> {
    let mut vec: Vec<f64> = Vec::new();
    let mut total: f64 = 0.0;
    for &x in xs {
        let val = x.exp();
        total += val;
        vec.push(val);
    }
    vec.iter_mut().map(|x| *x / total).collect()
}
