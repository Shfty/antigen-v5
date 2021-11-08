use std::time::{Duration, Instant};

pub fn spin_loop(duration: Duration, mut f: impl FnMut()) -> ! {
    loop {
        let ts = Instant::now();
        f();
        while Instant::now().duration_since(ts) < duration {
            std::hint::spin_loop();
        }
    }
}

