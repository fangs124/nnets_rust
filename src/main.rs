#![allow(dead_code)]
#![allow(clippy::needless_return)]
mod gameboards;
mod nnets;
mod train_nnets;
use crate::gameboards::*;
use crate::nnets::*;
use crate::train_nnets::*;
use crossterm::terminal::{Clear, ClearType};
use crossterm::{cursor, ExecutableCommand};
use inquire::Select;
use rand::Rng;
//use rayon::prelude::*;
use std::env;
use std::fmt;
use std::fmt::Display;
use std::fs::File;
#[allow(unused_imports)]
use std::io::{stderr, stdout, BufReader, BufWriter, Read, Write};
use std::sync::Arc;
use std::sync::Mutex;
use std::thread;
use std::time::Instant;
use windows::Win32::UI::Input::KeyboardAndMouse::GetAsyncKeyState;

//Choice State
#[derive(PartialEq, Eq, Clone)]
enum ChoiceState {
    NoChoice,
    Play,
    FinishedGame,
    Train,
    Sample,
    Quit,
}

type CS = ChoiceState;

impl Display for CS {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            CS::NoChoice => f.write_str("NoChoice"),
            CS::Play => f.write_str("Play"),
            CS::Train => f.write_str("Train"),
            CS::FinishedGame => f.write_str("FinishedGame"),
            CS::Sample => f.write_str("Sample"),
            CS::Quit => f.write_str("Quit"),
        }
    }
}

// some constants/parameters
const BATCH_SIZE: usize = 100;

fn main() -> std::io::Result<()> {
    /* debug related */
    env::set_var("RUST_BACKTRACE", "1");

    /* io related variables */
    let stdout = stdout();
    let mut stream_out: BufWriter<&std::io::Stdout> = BufWriter::new(&stdout);
    //let mut stream_err: BufWriter<&std::io::Stderr> = BufWriter::new(&stderr());

    let mut rng = rand::thread_rng();
    let mut choice: CS = CS::NoChoice;
    let start_q = String::from("NNRL v0.1:");
    let start_opt: Vec<CS> = vec![
        CS::Play,
        CS::Train,
        CS::Sample,
        CS::Quit,
        /* Sample Game,  Save NNet, Load NNet, Print NNet -- to add later?*/
    ];

    let mut is_quit = false;

    /* network loading/generating */
    let network_q = String::from("New Network?");
    let network_opt = vec!["New Network", "Load Network"];
    let node_count: Vec<usize> = vec![128, 128, 128, 128, 9]; //param_here
    let mut gb = GameBoard::NEW_GAMEBOARD;
    let mut network: Network<bool>;

    match Select::new(&network_q, network_opt).prompt().unwrap() {
        "New Network" => {
            let now = Instant::now();
            writeln!(stream_out, "Initializing...\n")?;
            network = Network::new_default(node_count.clone(), gb.to_vec_bool_x());
            writeln!(stream_out, "Elapsed: {:.2?}\n", now.elapsed())?;
        }
        "Load Network" => {
            let now = Instant::now();
            writeln!(stream_out, "Loading...\n")?;
            let file = File::open(format!("{:?}network.json", node_count))?;
            let mut buf_reader = BufReader::new(file);
            let mut contents = String::new();
            buf_reader.read_to_string(&mut contents)?;
            network = serde_json::from_str(&contents).unwrap();
            writeln!(stream_out, "Elapsed: {:.2?}\n", now.elapsed())?;
        }
        _ => {
            panic!("match error!");
        }
    }

    /* main loop */
    while !is_quit {
        match choice {
            CS::NoChoice => choice = Select::new(&start_q, start_opt.clone()).prompt().unwrap(),

            CS::Quit => {
                let quit_q = "Save Network?";
                let quit_opt = vec!["Yes", "No"];
                match Select::new(quit_q, quit_opt).prompt().unwrap() {
                    "Yes" => {
                        // save network
                        let file: File =
                            File::create(format!("{:?}network.json", node_count.clone()))?;
                        serde_json::to_writer(file, &network).unwrap();
                    }
                    "No" => return Ok(()),
                    _ => panic!("match error!"),
                }
                is_quit = true;
                continue;
            }

            CS::Train => {
                let mut thread_spawned: usize = 0;
                let mut thread_handles = Vec::new();
                let mut sb = SB::new();
                let mut is_on_policy: bool = rng.gen();
                /* training flow controls */
                let mut is_training = true;
                let mut is_playing_self = false;
                let mut loop_counter: usize = 0;
                let f = File::create(format!("{:?}log.txt", node_count.clone())).unwrap();
                let mut f_buff: BufWriter<&File> = BufWriter::new(&f);
                writeln!(stream_out, "Press q to stop.\n")?;
                sb.write_to_buf(&mut stream_out)?;
                let mut grad_vec: Arc<Mutex<Vec<Grad>>> = Arc::new(Mutex::new(Vec::new()));
                while is_training {
                    // listen to 'q' for interupt
                    let return_val = unsafe { GetAsyncKeyState(0x51_i32) };
                    if return_val & 0x01 != 0 {
                        //stop training
                        //choice = CS::NoChoice; //
                        is_training = false;
                    };
                    if thread_spawned <= 4 {
                        thread_spawned += 1;
                        if is_playing_self {
                            thread_handles.push(thread::spawn(move || {
                                grad_vec.lock().unwrap().append(&mut net_vs_self(
                                    network.clone(),
                                    &mut sb,
                                    true,
                                    false,
                                    is_on_policy,
                                ))
                            }));
                        } else {
                            thread_handles.push(thread::spawn(move || {
                                grad_vec.lock().unwrap().append(&mut net_vs_random(
                                    network.clone(),
                                    &mut sb,
                                    true,
                                    false,
                                    is_on_policy,
                                ))
                            }));
                        }
                    }

                    is_on_policy = rng.gen();
                    is_playing_self = rng.gen();
                    loop_counter += 1;
                    if loop_counter >= BATCH_SIZE {
                        network.update_sum(&grad_vec);
                        grad_vec = Vec::new();
                    }
                    if loop_counter >= BATCH_SIZE * 1000 {
                        sb.sample_output = network.get_pi_output();
                        //100K
                        sb.epoch += 1;
                        stream_out.execute(cursor::MoveUp(2)).unwrap();
                        stream_out
                            .execute(Clear(ClearType::FromCursorDown))
                            .unwrap();
                        sb.write_to_buf(&mut stream_out)?;
                        sb.write_to_buf(&mut f_buff)?;

                        sb.update();
                        loop_counter = 0;
                    }

                    if !is_training {
                        choice = CS::NoChoice
                    };
                }
            }
            CS::Sample => {
                let mut sb = SB::new();
                if let Ok(ans) =
                    Select::new("random or self play?", vec!["random", "self play"]).prompt()
                {
                    match ans {
                        "random" => {
                            net_vs_random(network.clone(), &mut sb, false, true, false);
                        }
                        "self play" => {
                            net_vs_self(network.clone(), &mut sb, false, true, false);
                        }
                        _ => panic!("prompt error!"),
                    }
                } else {
                    panic!("prompt error!");
                }
                choice = CS::NoChoice;
            }

            CS::Play => {
                net_vs_player(&mut network, &mut gb, &mut stream_out)?;
                if let Ok(ans) = Select::new("play again?", vec!["Yes", "No"]).prompt() {
                    match ans {
                        "Yes" => continue,
                        "No" => choice = CS::NoChoice,
                        _ => panic!("prompt error!"),
                    }
                } else {
                    panic!("prompt error!");
                }
            }
            _ => panic!("choice state impossible"),
        }
    }
    Ok(())
}

fn net_vs_player(
    net: &mut Network<bool>,
    gb: &mut GameBoard,
    stream: &mut BufWriter<&std::io::Stdout>,
) -> std::io::Result<()> {
    gb.new_game();
    let mut rng = rand::thread_rng();
    let net_is_x: bool = rng.gen();
    let mut x_turn: bool = true; // x goes first
    let move_q = String::from("Please select a square.");

    while gb.game_state() == GS::Ongoing {
        #[allow(non_snake_case)]
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
            let index: usize = get_random_index(&output, gb);
            gb.make_move(BB::MOVES[index])
                .expect("net_vs_player: invalid move");
            writeln!(stream, "net's turn: ")?;
        }
        // player's turn
        else {
            // make a random valid move
            let indices: Vec<usize> = (0..9)
                .filter(|&i| gb.is_valid_move(&BB::MOVES[i]))
                .collect();
            let choice = Select::new(&move_q, indices).prompt().unwrap();
            writeln!(stream, "player's turn:")?;
            gb.make_move(BB::MOVES[choice]).unwrap();
        }
        gb.print_gameboard();
        // pass turn to next player
        x_turn = !x_turn;
    }

    if gb.game_state() == GS::Tie {
        writeln!(stream, "a draw!")?;
    } else if (gb.game_state() == GS::XWin) == net_is_x {
        writeln!(stream, "net wins!")?;
    } else {
        writeln!(stream, "player wins!")?;
    }
    Ok(())
}
