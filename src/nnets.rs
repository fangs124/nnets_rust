#![allow(dead_code)]
extern crate nalgebra as na;

use na::base::{DMatrix, DVector};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
pub struct Network<T> {
    pub node_count: Vec<usize>,
    pub input_data: Vec<T>,
    pub alpha: f64,                  // learning rate
    pub gamma: f64,                  // patience factor
    weights: Vec<Box<DMatrix<f64>>>, // row only (col vector)
    biases: Vec<Box<DVector<f64>>>,  // row,col
    layers: Vec<Box<DVector<f64>>>,  // row,col //stores z_j as in a_j = phi(z_j)
    dw: Vec<Box<DMatrix<f64>>>,
    db: Vec<Box<DVector<f64>>>,
    ty: PhiT, // to select internal activation function used
}

fn phi(ty: &PhiT) -> fn(f64) -> f64 {
    match ty {
        PhiT::Sigmoid => sigmoid,
        PhiT::ReLU => relu,
        PhiT::LReLU => lrelu,
        PhiT::Tanh => tanh,
    }
}

fn dphi(ty: &PhiT) -> fn(f64) -> f64 {
    match ty {
        PhiT::Sigmoid => dsigmoid,
        PhiT::ReLU => drelu,
        PhiT::LReLU => dlrelu,
        PhiT::Tanh => dtanh,
    }
}

#[derive(Serialize, Deserialize)]
enum PhiT {
    Sigmoid,
    ReLU,
    LReLU,
    Tanh,
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
    1 as f64 / (1 as f64 + std::f64::consts::E.powf(-x))
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
    1.0 - (x.tanh() * x.tanh())
}

const DEFAULT_ALPHA: f64 = 1.00;
const DEFAULT_GAMMA: f64 = 0.95;
const DEFAULT_PHI: PhiT = PhiT::Sigmoid;
// idk how to implement partial defaults, so document here
// alpha = 0.05, gamma = 0.95

impl<T: InputType> Network<T> {
    //internal node count describes internal layer
    pub fn new_default(internal_nodes: Vec<usize>, input_data: Vec<T>) -> Self {
        let mut weights: Vec<Box<DMatrix<f64>>> = Vec::with_capacity(internal_nodes.len());
        let mut biases: Vec<Box<DVector<f64>>> = Vec::with_capacity(internal_nodes.len() + 1);
        let mut layers: Vec<Box<DVector<f64>>> = Vec::with_capacity(internal_nodes.len() + 1);
        let mut dw: Vec<Box<DMatrix<f64>>> = Vec::with_capacity(internal_nodes.len() - 1);
        let mut db: Vec<Box<DVector<f64>>> = Vec::with_capacity(internal_nodes.len());
        let mut j: usize = input_data.len(); //previous node count
        let mut node_count: Vec<usize> = vec![internal_nodes.len()];
        weights.push(Box::new(DMatrix::new_random(j, j)));
        dw.push(Box::new(DMatrix::zeros(j, j))); // M[i]: layer[j] -> layer[i]
        biases.push(Box::new(DVector::new_random(j)));
        db.push(Box::new(DVector::zeros(j)));
        layers.push(Box::new(DVector::zeros(j)));
        for i in internal_nodes {
            weights.push(Box::new(DMatrix::new_random(i, j))); // M[i]: layer[j] -> layer[i]
            biases.push(Box::new(DVector::new_random(i)));
            layers.push(Box::new(DVector::zeros(i)));
            dw.push(Box::new(DMatrix::zeros(i, j))); // M[i]: layer[j] -> layer[i]
            db.push(Box::new(DVector::zeros(i)));
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
            db,
            dw,
            ty: DEFAULT_PHI,
        };
    }

    pub fn set_input(&mut self, input_data: Vec<T>) {
        self.input_data = input_data;
    }

    pub fn get_z_output_(&mut self, input_data: &mut Vec<T>) -> Vec<f64> {
        self.forward_prop(input_data);
        self.layers[self.node_count.len() - 1]
            .data
            .as_vec()
            .to_vec()
    }

    pub fn get_phiz_output_(&mut self, input_data: &mut Vec<T>) -> Vec<f64> {
        self.forward_prop(input_data);
        self.layers[self.node_count.len() - 1]
            .map(phi(&self.ty))
            .data
            .as_vec()
            .to_vec()
    }

    pub fn get_pi_output(&mut self, input_data: &mut Vec<T>) -> Vec<f64> {
        self.forward_prop(input_data);
        softmax(
            self.layers[self.node_count.len() - 1]
                .map(phi(&self.ty))
                .data
                .as_vec(),
        )
    }

    pub fn forward_prop(&mut self, input_data: &mut Vec<T>) {
        // turn input data into usable format.
        let input_vector: DVector<f64> =
            DVector::from_iterator(input_data.len(), input_data.iter_mut().map(|x| x.to_f64()));

        // layer l = 0
        // z_l = W_l * phi(z_{l-1}) + b_l
        *self.layers[0] = &*self.weights[0] * input_vector.map(phi(&self.ty)) + &*self.biases[0];

        // layer l = 1..L-1
        for l in 1..self.node_count.len() {
            // z_l = W_l * phi(z_{l-1}) + b_l
            *self.layers[l] =
                &*self.weights[l] * (&*self.layers[l - 1]).map(phi(&self.ty)) + &*self.biases[l];
        }
    }

    // computes averaged out dpi/dtheta for policy pi_theta iteration.
    pub fn train(&mut self, train_set: Vec<(Vec<T>, Vec<f64>, usize)>) {
        let gamma: f64 = self.gamma.powi(train_set.len() as i32);
        for (mut input_data, reward, index) in train_set.into_iter() {
            self.back_prop(&mut input_data, &reward, gamma, index);
        }
    }
    pub fn train_iter(
        &mut self,
        train_set: impl Iterator<Item = (Vec<T>, Vec<f64>, usize)>,
        len: usize,
    ) {
        let gamma: f64 = self.gamma.powi(len as i32);
        for (mut input_data, reward, index) in train_set {
            self.back_prop(&mut input_data, &reward, gamma, index);
        }
    }

    // dCda in the function argument is more like the reward
    // pi = softmax(phi(z))
    // dpi/da = softmax'(phi(z))
    pub fn back_prop(&mut self, input: &mut Vec<T>, reward: &Vec<f64>, gamma: f64, index: usize) {
        let input_vector = DVector::from_iterator(input.len(), input.iter().map(|x| x.to_f64()));
        let index = index;
        let pi_vec = self.get_pi_output(input); // probability output softmax(phi(z))
        let pi_index = pi_vec[index]; // pi(a|s) corresponding to the taken action a
        let mut dpida_vec: Vec<f64> = Vec::new(); // gradient of pi_index wrt to network output a = phi(z)
        for i in 0..reward.len() {
            if i == index {
                dpida_vec.push(pi_index * (1.0 - pi_index));
            } else {
                dpida_vec.push(-1.0 * pi_vec[i] * pi_index);
            }
        }
        let mut dpida: DVector<f64> = DVector::from_vec(dpida_vec);
        let layer_count = self.layers.len();
        for l in 1..=layer_count {
            // dpi/dz = dpi/da * da/dz
            //        = dpi/da * dphi(z)
            let z = &self.layers[layer_count - l];
            let dpidz = dpida.component_mul(&z.map(dphi(&self.ty)));

            // dpi/db = dpi/da * da/dz * dz/db
            //        =(dpi/dc)        * 1
            *self.db[layer_count - l] =
                &*self.db[layer_count - l] + (gamma * reward[index] / pi_index) * &dpidz;

            if l != layer_count {
                // dpi/dw = dpi/da * da/dz   * dz/dw
                //        = dpi/da * phi'(z) * phi(z)
                // also note that:   (AB)^t = B^t A^t,
                //             so: (AB^t)^t = B A^t, <-- what we have here
                *self.dw[layer_count - l] = &*self.dw[layer_count - l]
                    + (gamma * reward[index] / pi_index)
                        * &dpidz
                        * (&self.layers[layer_count - (l + 1)].map(phi(&self.ty))).transpose();
            } else {
                // same as above, except we use input vector
                *self.dw[layer_count - l] = &*self.dw[layer_count - l]
                    + (gamma * reward[index] / pi_index)
                        * &dpidz
                        * (&input_vector.map(phi(&self.ty))).transpose();
            }

            // dpi/da for the next layer:
            // dpi/da'= sum dpi/da *   da/dz * dz/da'
            //        = sum  dz/da'* (dpi/da * da/dz)
            //        =        [w] * [dpi/da * phi'(z)]
            dpida = self.weights[layer_count - l].tr_mul(&dpidz);
        }
    }

    // pushes policy interation theta = theta + alpha * dpi/dtheta (gamma was factored into the dw and db calculation)
    pub fn update(&mut self) {
        let layer_count = self.layers.len();
        for i in 0..layer_count {
            *self.weights[i] = &*self.weights[i] + self.alpha * &*self.dw[i];
            (*self.dw)[i].fill(0.0);
            *self.biases[i] = &*self.biases[i] + self.alpha * &*self.db[i];
            (*self.db)[i].fill(0.0);
        }
    }
}

pub trait InputType {
    fn to_f64(&self) -> f64;
}

pub fn softmax(xs: &Vec<f64>) -> Vec<f64> {
    let mut vec: Vec<f64> = Vec::new();
    let mut total: f64 = 0.0;
    for &x in xs {
        let val = std::f64::consts::E.powf(x);
        total += val;
        vec.push(val);
    }
    vec.iter_mut().map(|x| *x / total).collect()
}
