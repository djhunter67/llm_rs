use log::error;
use rust_bert::{
    gpt2::{Gpt2ConfigResources, Gpt2MergesResources, Gpt2ModelResources, Gpt2VocabResources},
    pipelines::{
        common::{ModelResource, ModelType},
        text_generation::{TextGenerationConfig, TextGenerationModel},
    },
    resources::RemoteResource,
};

fn main() {
    let cuda = tch::Cuda::is_available();

    println!("\nCUDA?: {cuda}\n");

    let cuda_info = tch::Cuda::device_count();

    println!("\nNUM OF CUDA DEVICES: {cuda_info}\n");

    // let more_cuda_info = tch::Device::Cuda(1);

    let more_cuda_info = tch::Device::Cuda(0);

    println!("\nMORE CUDA INFO: {more_cuda_info:?}\n");

    // let model_resource = Box::new(LocalResource {
    // local_path: PathBuf::from("/home/djhunter67/Downloads/model_gpt2.ot"),
    // });

    let model_resource = Box::new(RemoteResource::from_pretrained(Gpt2ModelResources::GPT2));

    let config_resource = Box::new(RemoteResource::from_pretrained(Gpt2ConfigResources::GPT2));

    let vocab_resource = Box::new(RemoteResource::from_pretrained(Gpt2VocabResources::GPT2));

    let merges_resource = Box::new(RemoteResource::from_pretrained(Gpt2MergesResources::GPT2));

    let generation_config = TextGenerationConfig {
        // model_type: ModelType::GPTNeo,
        model_type: ModelType::GPT2,
        model_resource: ModelResource::Torch(model_resource.clone()),
        config_resource: config_resource.clone(),
        vocab_resource: vocab_resource.clone(),
        merges_resource: Some(merges_resource.clone()),
        num_beams: 2,
        no_repeat_ngram_size: 2,
        max_length: Some(500),
        device: tch::Device::Cuda(0),
        // device: tch::Device::Cpu,
        ..Default::default()
    };

    let model = match TextGenerationModel::new(generation_config) {
        Ok(model) => model,
        Err(e) => {
            println!("Due to: {e}\nFalling back to CPU...");
            TextGenerationModel::new(TextGenerationConfig {
                // model_type: ModelType::GPTNeo,
                model_type: ModelType::GPTJ,
                model_resource: ModelResource::Torch(model_resource),
                config_resource,
                vocab_resource,
                merges_resource: Some(merges_resource),
                num_beams: 2,
                no_repeat_ngram_size: 2,
                max_length: Some(500),
                // device: tch::Device::Cpu,
                device: tch::Device::Cuda(1),
                ..Default::default()
            })
            .unwrap_or_else(|_| {
                error!("Unable to process error");
                // exit the program
                std::process::exit(1);
            })
        }
    };

    loop {
        println!("\nEnter a prompt: \n");
        let mut line = String::new();
        std::io::stdin().read_line(&mut line).unwrap();
        println!("\nProcessing...\n\n");

        let split = line.split('/').collect::<Vec<&str>>();

        let slc = split.as_slice();

        let output = model.generate(&slc[1..], Some(slc[0])).unwrap();

        for sentence in output {
            println!("{sentence}");
        }
    }
}
