#![allow(dead_code)]
use itertools::interleave;
use rand::distributions::{Distribution, WeightedIndex};
use rand::Rng;

use crate::gameboards::*;
use crate::nnets::*;

use std::io::{BufWriter, Write};
use std::time::Instant;

#[derive(PartialEq, Copy, Clone)]
pub struct ScoreBoard {
    pub invalid_count: u64,
    pub vs_random: u64,
    pub vs_self: u64,
    pub net_wins: u64,
    pub random_wins: u64,
    pub draws: u64,
    pub self_plays: u64,
    pub self_draws: u64,
    pub prev_w: f64,
    pub prev_l: f64,
    pub prev_d: f64,
    pub epoch: u32,
    pub start_time: Instant,
    pub now: Instant,
}

pub type SB = ScoreBoard;
pub type Net<T> = Network<T>;

impl ScoreBoard {
    pub fn new() -> Self {
        ScoreBoard {
            invalid_count: 0,
            vs_random: 0,
            vs_self: 0,
            net_wins: 0,
            random_wins: 0,
            draws: 0,
            self_plays: 0,
            self_draws: 0,
            prev_w: 0.0,
            prev_l: 0.0,
            prev_d: 0.0,
            epoch: 0,
            start_time: Instant::now(),
            now: Instant::now(),
        }
    }
    pub fn update(&mut self) {
        self.prev_w = 100.0 * (self.net_wins as f64) / (self.vs_random as f64);
        self.prev_l = 100.0 * (self.random_wins as f64) / (self.vs_random as f64);
        self.prev_d = 100.0 * (self.draws as f64) / (self.vs_random as f64);
        self.self_plays = 0;
        self.net_wins = 0;
        self.random_wins = 0;
        self.draws = 0;
        self.vs_self = 0;
        self.vs_random = 0;
        self.self_draws = 0;
        self.invalid_count = 0;
        self.now = Instant::now();
    }

    pub fn write_to_buf<T: Write>(&mut self, stream: &mut BufWriter<T>) -> std::io::Result<()> {
        writeln!(
            stream,
            "N wins: {}, R wins: {}, Draws: {}, [{:.2}+({:.2}):{:.2}+({:.2}):{:.2}+({:.2})] ({}k-epoch: {}, {:.2?}) invalid this epoch: {}",
            self.net_wins,
            self.random_wins,
            self.draws,
            100.0 * (self.net_wins as f64) / (self.vs_random as f64),
            (100.0 * (self.net_wins as f64) / (self.vs_random as f64)) - self.prev_w,
            100.0 * (self.random_wins as f64) / (self.vs_random as f64),
            (100.0 * (self.random_wins as f64) / (self.vs_random as f64)) - self.prev_l,
            100.0 * (self.draws as f64) / (self.vs_random as f64),
            (100.0 * (self.draws as f64) / (self.vs_random as f64)) - self.prev_d,
            BATCH_SIZE,
            self.epoch,
            self.now.elapsed(),
            self.invalid_count,
        )?;

        writeln!(
            stream,
            "Time Training: {:.2?}, Self Play: {} (Draws: {:.2}%)",
            self.start_time.elapsed(),
            self.self_plays,
            (self.self_draws as f64 / self.self_plays as f64) * 100.0,
        )?;
        stream.flush()?;
        Ok(())
    }
}

impl BitBoard {
    pub fn to_vec_bool(&self) -> Vec<bool> {
        let mut vec: Vec<bool> = Vec::new();
        for i in 0..=11 {
            if i % 4 != 0 {
                vec.push(self.get_val() & (1 << (11 - i)) != 0);
            }
        }
        vec
    }
}

impl GB {
    pub fn to_vec_bool_x(&self) -> Vec<bool> {
        let mut vec: Vec<bool> = Vec::new();
        vec.append(&mut self.x_b.to_vec_bool());
        vec.append(&mut self.o_b.to_vec_bool());
        vec
    }
    pub fn to_vec_bool_o(&self) -> Vec<bool> {
        let mut vec: Vec<bool> = Vec::new();
        vec.append(&mut self.o_b.to_vec_bool());
        vec.append(&mut self.x_b.to_vec_bool());
        vec
    }
}

impl InputType for bool {
    fn to_f64(&self) -> f64 {
        match self {
            true => 1.0,
            false => 0.0,
        }
    }
}

pub struct Episode<T> {
    //a sequence of data from start till end of game
    states: Vec<Vec<T>>,
    actions: Vec<usize>,
    rewards: Vec<f64>,
}

type Ep<T> = Episode<T>;

impl<T> Episode<T> {
    pub fn new() -> Self {
        Episode {
            states: Vec::new(),
            actions: Vec::new(),
            rewards: Vec::new(),
        }
    }
}
// some constants/parameters
const BATCH_SIZE: usize = 100;
const TRESHOLD: f64 = 0.001;
pub fn net_vs_self(
    net: &mut Net<bool>,
    gb: &mut GB,
    sb: &mut SB,
    is_train: bool,
    sample: bool,
    is_on_policy: bool,
) {
    if sample {
        println!("net_vs_self");
    }

    gb.new_game();

    let mut x_turn: bool = true; // x goes first

    let mut grad_vec: Vec<Grad> = Vec::new();
    let mut rewards_x: Vec<f64> = Vec::new();
    let mut rewards_o: Vec<f64> = Vec::new();
    let mut inv_grad_vec: Vec<Grad> = Vec::new();
    let mut inv_rewards: Vec<f64> = Vec::new();

    while gb.game_state() == GS::Ongoing {
        let mut vec_bool: Vec<bool> = Vec::new();
        if x_turn {
            vec_bool.append(&mut gb.to_vec_bool_x());
        } else {
            vec_bool.append(&mut gb.to_vec_bool_o());
        }

        net.forward_prop(&vec_bool);
        let output: Vec<f64> = net.get_pi_output();

        // teach Net not to make invalid moves
        let inv_indices = get_invalid_indices(&output, TRESHOLD, gb);

        for &i in &inv_indices {
            inv_grad_vec.push(net.back_prop(&vec_bool, &i));
            inv_rewards.push(-2.5);
        }

        sb.invalid_count += inv_indices.len() as u64;

        let index: usize = match is_on_policy {
            true => get_random_index(&output, gb),
            false => get_index(&output, gb),
        };
        gb.make_move(BB::MOVES[index]).unwrap();
        grad_vec.push(net.back_prop(&vec_bool, &index));
        if sample {
            gb.print_gameboard();
        }

        if x_turn {
            rewards_x.push(1.0);
        } else {
            rewards_o.push(1.0);
        }
        // pass turn to next player
        x_turn = !x_turn;
    }

    // game ended
    if is_train {
        let mut rewards: Vec<f64> = Vec::new();
        match gb.game_state() {
            GS::XWin => {
                rewards = interleave(rewards_x, rewards_o.into_iter().map(|x| x * -1.0))
                    .collect::<Vec<_>>();
            }
            GS::OWin => {
                rewards = interleave(rewards_x.into_iter().map(|x| x * -1.0), rewards_o)
                    .collect::<Vec<_>>();
            }
            GS::Tie => sb.self_draws += 1,
            _ => panic!("net_vs_random: state is impossible"),
        };
        let inv_len = inv_grad_vec.len();
        let len = grad_vec.len();
        if inv_len > 0 {
            net.update(&inv_grad_vec, &vec![1_i32; inv_len], &inv_rewards);
        }
        if gb.game_state() != GS::Tie {
            net.update(&grad_vec, &vec![len as i32; len], &rewards);
        }
    }
    sb.self_plays += 1;
}

pub fn net_vs_random(
    net: &mut Net<bool>,
    gb: &mut GB,
    sb: &mut SB,
    is_train: bool,
    sample: bool,
    is_on_policy: bool,
) {
    let mut rng = rand::thread_rng();
    let net_is_x: bool = rng.gen();

    if sample {
        println!("net_vs_random");
        println!("net is x: {}", net_is_x);
    }

    gb.new_game();

    let mut x_turn: bool = true; // x goes first

    let mut grad_vec: Vec<Grad> = Vec::new();
    let mut rewards_x: Vec<f64> = Vec::new();
    let mut rewards_o: Vec<f64> = Vec::new();
    let mut inv_grad_vec: Vec<Grad> = Vec::new();
    let mut inv_rewards: Vec<f64> = Vec::new();

    while gb.game_state() == GS::Ongoing {
        let mut vec_bool: Vec<bool> = Vec::new();

        if x_turn {
            vec_bool.append(&mut gb.to_vec_bool_x());
        } else {
            vec_bool.append(&mut gb.to_vec_bool_o());
        }

        // net's turn
        if x_turn == net_is_x {
            net.forward_prop(&vec_bool);
            let output: Vec<f64> = net.get_pi_output();

            // teach Net not to make invalid moves
            let inv_indices = get_invalid_indices(&output, TRESHOLD, gb);

            for &i in &inv_indices {
                inv_grad_vec.push(net.back_prop(&vec_bool, &i));
                inv_rewards.push(-1.0);
            }

            sb.invalid_count += inv_indices.len() as u64;

            let index: usize = match is_on_policy {
                true => get_random_index(&output, gb),
                false => get_index(&output, gb),
            };
            gb.make_move(BB::MOVES[index]).unwrap();
            grad_vec.push(net.back_prop(&vec_bool, &index));
            if sample {
                gb.print_gameboard();
            }
            if x_turn {
                rewards_x.push(1.0);
            } else {
                rewards_o.push(1.0);
            }
        }
        // random's turn
        else {
            // make a random valid move
            let indices: Vec<usize> = (0..9)
                .filter(|&i| gb.is_valid_move(&BB::MOVES[i]))
                .collect();
            let rand_num = rng.gen_range(0..indices.len());
            gb.make_move(BB::MOVES[indices[rand_num]]).unwrap();

            if sample {
                gb.print_gameboard();
            }

            net.forward_prop(&vec_bool);
            let output = net.get_pi_output();
            if output[indices[rand_num]].abs() > TRESHOLD {
                if x_turn {
                    rewards_x.push(1.0);
                } else {
                    rewards_o.push(1.0);
                }
            } else if x_turn {
                rewards_x.push(0.0);
            } else {
                rewards_o.push(0.0);
            }
        }

        // pass turn to next player
        x_turn = !x_turn;
    }
    // game ended

    //update score statistics
    sb.vs_random += 1;
    if gb.game_state() == GS::Tie {
        sb.draws += 1
    } else if (gb.game_state() == GS::XWin) == net_is_x {
        sb.net_wins += 1
    } else {
        sb.random_wins += 1
    }

    if is_train {
        let mut rewards: Vec<f64> = Vec::new();
        match gb.game_state() {
            GS::XWin => {
                rewards = interleave(rewards_x, rewards_o.into_iter().map(|x| x * -1.0))
                    .collect::<Vec<_>>();
            }
            GS::OWin => {
                rewards = interleave(rewards_x.into_iter().map(|x| x * -1.0), rewards_o)
                    .collect::<Vec<_>>();
            }
            GS::Tie => sb.self_draws += 1,
            _ => panic!("net_vs_random: state is impossible"),
        };
        let inv_len = inv_grad_vec.len();
        let len = grad_vec.len();
        if inv_len > 0 {
            net.update(&inv_grad_vec, &vec![1_i32; inv_len], &inv_rewards);
        }
        if gb.game_state() != GS::Tie {
            net.update(&grad_vec, &vec![len as i32; len], &rewards);
        }
    }
}

// returns the indices of all invalid moves by the Net that's above the treshold
pub fn get_invalid_indices(net_output: &[f64], treshold: f64, gb: &GB) -> Vec<usize> {
    net_output
        .iter()
        .enumerate()
        .filter(|&(i, &x)| x.abs() > treshold && !gb.is_valid_move(&BB::MOVES[i]))
        .map(|(index, _)| index)
        .collect()
}

// returns the index of a valid move corresponding to the Net's highest output
pub fn get_index(net_output: &[f64], gb: &GB) -> usize {
    net_output
        .iter()
        .enumerate()
        .filter(|&(i, _)| gb.is_valid_move(&BB::MOVES[i]))
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(index, _)| index)
        .unwrap()
}

pub fn get_random_index(net_output: &[f64], gb: &GB) -> usize {
    let mut valids = net_output
        .iter()
        .enumerate()
        .filter(|&(i, &x)| gb.is_valid_move(&BB::MOVES[i]) && x.abs() > 0.00001);
    let mut total: f64 = 0.0;
    for (_, &x) in valids.clone() {
        total += x
    }
    let ws = valids
        .clone()
        .map(|(_, &x)| x / total)
        .collect::<Vec<f64>>();
    let mut rng = rand::thread_rng();
    let dist = WeightedIndex::new(ws).unwrap();

    return valids.nth(dist.sample(&mut rng)).unwrap().0;
}
