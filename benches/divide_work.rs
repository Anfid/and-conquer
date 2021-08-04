use and_conquer::{divide_equal_work, divide_work};
use criterion::{black_box, criterion_group, criterion_main, BatchSize, Criterion};

fn bench_small_equal_work(c: &mut Criterion) {
    let input: Vec<i32> = (1..1000).collect();
    let mut group = c.benchmark_group("small_equal_work");
    group.bench_with_input("vec mtx", &input, |b, i| {
        b.iter_batched(
            || i.clone(),
            |i| divide_work(black_box(i), black_box(|x| x * 2)),
            BatchSize::SmallInput,
        )
    });
    group.bench_with_input("equal", &input, |b, i| {
        b.iter_batched(
            || i.clone(),
            |i| divide_equal_work(black_box(i), black_box(|x| x * 2)),
            BatchSize::SmallInput,
        )
    });
    group.finish();
}

fn fibonacci(n: u64) -> u64 {
    match n {
        0 => 1,
        1 => 1,
        n => fibonacci(n - 1) + fibonacci(n - 2),
    }
}

fn bench_unequal_work(c: &mut Criterion) {
    let input: Vec<u64> = (0..30).collect();
    let mut group = c.benchmark_group("unequal_work");
    group.bench_with_input("vec mtx", &input, |b, i| {
        b.iter_batched(
            || i.clone(),
            |i| divide_work(black_box(i), black_box(fibonacci)),
            BatchSize::SmallInput,
        )
    });
    group.bench_with_input("equal", &input, |b, i| {
        b.iter_batched(
            || i.clone(),
            |i| divide_equal_work(black_box(i), black_box(fibonacci)),
            BatchSize::SmallInput,
        )
    });
    group.finish();
}

criterion_group!(benches, bench_small_equal_work, bench_unequal_work,);
criterion_main!(benches);
