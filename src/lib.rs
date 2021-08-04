use std::sync::{Arc, Mutex};

// For practical purposes should probably be larger
const PARALLEL_WORK_THRESHOLD: usize = 10;

/// Splits work between threads if amount of elements in `input` is greater than or equal to
/// `PARALLEL_WORK_THRESHOLD = 10`. This function is better suitable for tiny or equal chunks of work regardless of
/// input value. If computational time required to complete `f` varies greatly from input values, [divide_work]
/// works better.
pub fn divide_equal_work<F, T, R>(mut input: Vec<T>, f: F) -> Vec<R>
where
    T: Send + 'static,
    R: Send + 'static,
    F: Fn(T) -> R + Send + Sync + Clone + 'static,
{
    if input.len() < PARALLEL_WORK_THRESHOLD {
        input.into_iter().map(f).collect()
    } else {
        let cores = num_cpus::get();
        let length = input.len();
        let tasks_per_worker = length as f32 / cores as f32;

        let workers = (0..cores)
            .rev()
            .map(|core| input.split_off((tasks_per_worker * core as f32).round() as usize))
            .map(|tasks| {
                let f = f.clone();
                std::thread::spawn(move || tasks.into_iter().map(f).collect::<Vec<_>>())
            })
            // Collect is required to actually spawn threads
            .collect::<Vec<_>>();

        // Workers are grouped in reversed order, but the value they return preserves original vector order. Joining their
        // produced output is enough if order of workers is reversed back.
        workers
            .into_iter()
            .rev()
            .map(|w| {
                w.join()
                    // If function `f` panics, we should panic too, so that output vector has results for all input
                    // values.
                    .unwrap_or_else(|_| panic!("Worker thread panicked"))
                    .into_iter()
            })
            .flatten()
            .collect()
    }
}

/// Splits work between threads if amount of elements in `input` is greater than or equal to
/// `PARALLEL_WORK_THRESHOLD = 10`. This function schedules work evenly between each thread, but scheduling comes
/// with extra overhead. If work required to complete `f` is expected to be equal regardless of input value, it is
/// best to use [divide_equal_work].
pub fn divide_work<F, T, R>(input: Vec<T>, f: F) -> Vec<R>
where
    T: Send + 'static,
    R: Send + 'static,
    F: Fn(T) -> R + Send + Sync + Clone + 'static,
{
    if input.len() < PARALLEL_WORK_THRESHOLD {
        input.into_iter().map(f).collect()
    } else {
        let cores = num_cpus::get();
        let length = input.len();

        let workers = {
            let queue = Arc::new(Mutex::new(input));

            let mut workers = Vec::with_capacity(cores);
            for _ in 0..cores {
                let queue = queue.clone();
                let f = f.clone();

                workers.push(std::thread::spawn(move || {
                    let mut res = Vec::new();
                    loop {
                        let value = {
                            let mut q = queue.lock().unwrap();
                            let val = q.pop();
                            // At this point len already has element index, because it was decrememted with pop.
                            // `idx` will be unused if `pop` returns `None`
                            (q.len(), val)
                        };
                        if let (idx, Some(val)) = value {
                            // SAFETY: `idx` must remain within `length` to prevent writing data out of array bounds
                            res.push((idx, f(val)));
                        } else {
                            break;
                        }
                    }
                    res
                }));
            }

            workers
        };

        // SAFETY: capacity must be >= than used in `set_len`
        let mut res = Vec::with_capacity(length);
        let res_mut_ptr: *mut R = res.as_mut_ptr();

        for w in workers {
            match w.join() {
                Ok(worker_res) => {
                    for (idx, r) in worker_res {
                        // SAFETY: 1) idx is obtained from input Vec enumeration, input and output vectors have
                        //         the same length, so idx remains within bounds.
                        //         2) ptr is a valid location to write, because it is obtained from preallocated
                        //         vector with required capacity.
                        unsafe {
                            std::ptr::write(res_mut_ptr.add(idx), r);
                        }
                    }
                }
                // SAFETY: Necessary to prevent vector having uninitialized elements
                Err(_) => panic!("Worker thread panicked"),
            }
        }

        // SAFETY: 1) Allocated with capacity `length`
        //         2) Workers produce value for each element, initialization occurs after joining worker threads.
        //         In case function `f` panics, this statement is unreachable.
        unsafe { res.set_len(length) }

        res
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn divide_numbers() {
        let input = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
        let f = |x| x + 1;
        let output = divide_work(input, f);

        let expected = vec![2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13];
        assert_eq!(output, expected);
    }

    #[test]
    fn divide_equal_numbers() {
        let input = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
        let f = |x| x + 1;
        let output = divide_equal_work(input, f);

        let expected = vec![2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13];
        assert_eq!(output, expected);
    }

    #[test]
    fn divide_strings() {
        let input = vec![
            "walk", "show", "code", "enter", "etc", "etc", "etc", "etc", "etc", "etc", "etc", "etc",
        ];
        let tail = String::from("ed");
        let output = divide_work(input, move |verb| String::from(verb) + &tail);

        let expected: Vec<String> = vec![
            "walked", "showed", "codeed", "entered", "etced", "etced", "etced", "etced", "etced",
            "etced", "etced", "etced",
        ]
        .into_iter()
        .map(|verb| String::from(verb))
        .collect();

        assert_eq!(output, expected);
    }

    #[test]
    fn divide_equal_strings() {
        let input = vec![
            "walk", "show", "code", "enter", "etc", "etc", "etc", "etc", "etc", "etc", "etc", "etc",
        ];
        let tail = String::from("ed");
        let output = divide_equal_work(input, move |verb| String::from(verb) + &tail);

        let expected: Vec<String> = vec![
            "walked", "showed", "codeed", "entered", "etced", "etced", "etced", "etced", "etced",
            "etced", "etced", "etced",
        ]
        .into_iter()
        .map(|verb| String::from(verb))
        .collect();

        assert_eq!(output, expected);
    }

    #[test]
    fn divide_undivisible() {
        let input = vec!["walk", "show", "code", "enter"];

        // Ensure `input` is not qualified for splitting
        assert!(input.len() < PARALLEL_WORK_THRESHOLD);

        let tail = String::from("ed");
        let output = divide_work(input, move |verb| String::from(verb) + &tail);

        let expected: Vec<String> = vec!["walked", "showed", "codeed", "entered"]
            .into_iter()
            .map(|verb| String::from(verb))
            .collect();

        assert_eq!(output, expected);
    }

    #[test]
    fn divide_equal_undivisible() {
        let input = vec!["walk", "show", "code", "enter"];

        // Ensure `input` is not qualified for splitting
        assert!(input.len() < PARALLEL_WORK_THRESHOLD);

        let tail = String::from("ed");
        let output = divide_equal_work(input, move |verb| String::from(verb) + &tail);

        let expected: Vec<String> = vec!["walked", "showed", "codeed", "entered"]
            .into_iter()
            .map(|verb| String::from(verb))
            .collect();

        assert_eq!(output, expected);
    }
}
