use criterion::{black_box, criterion_group, criterion_main, Criterion};
use axiom::inference::sampler::{Sampler, SamplerConfig};

fn bench_sampler(c: &mut Criterion) {
    let mut group = c.benchmark_group("sampler");

    let logits = vec![0.1f32; 128256];
    let previous_tokens: Vec<u32> = (0..64).collect();

    group.bench_function("greedy_no_penalty_128k", |b| {
        let mut sampler = Sampler::new(SamplerConfig {
            temperature: 0.0,
            ..Default::default()
        });
        b.iter(|| {
            sampler.sample(black_box(&logits), black_box(&previous_tokens))
        });
    });

    group.bench_function("greedy_repetition_penalty_128k", |b| {
        let mut sampler = Sampler::new(SamplerConfig {
            temperature: 0.0,
            repetition_penalty: 1.2,
            ..Default::default()
        });
        b.iter(|| {
            sampler.sample(black_box(&logits), black_box(&previous_tokens))
        });
    });

    group.bench_function("greedy_no_repeat_ngram_128k", |b| {
        let mut sampler = Sampler::new(SamplerConfig {
            temperature: 0.0,
            no_repeat_ngram_size: Some(3),
            ..Default::default()
        });
        b.iter(|| {
            sampler.sample(black_box(&logits), black_box(&previous_tokens))
        });
    });

    group.finish();
}

criterion_group!(benches, bench_sampler);
criterion_main!(benches);

