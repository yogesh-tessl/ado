# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT
import json
import typing

import ray
from lm_eval.evaluator import simple_evaluate

from orchestrator.modules.actuators.measurement_queue import MeasurementQueue
from orchestrator.schema.experiment import Experiment, ParameterizedExperiment
from orchestrator.schema.property_value import PropertyValue
from orchestrator.schema.request import MeasurementRequest, MeasurementRequestStateEnum
from orchestrator.schema.result import ValidMeasurementResult
from orchestrator.utilities.support import get_experiment_input_values


# Execute an LLM eval harness experiment on a model served by a vllm endpoint
def llm_eval_harness_vllm_experiment(
    vllm_endpoint_url: str,
    batch_size: int,
    model_name: str,
    num_fewshot: int = 0,
    limit: int | None = None,
    seed: int | None = None,
    tasks: list[str] = ["arc_challenge"],
    **extra_args,
) -> dict[str, typing.Any]:
    """
    Run an LLM eval harness experiment on a vllm model

    This function uses the lm_eval library to run the experiment.
    https://github.com/EleutherAI/lm-evaluation-harness

    Args:
        vllm_endpoint_url: The URL of the vllm endpoint to use for inference.
        model_name: The name of the model to use for evaluation.
        batch_size: The batch size to use for evaluation.
        num_fewshot: The number of few-shot examples to use. Default is 0.
        limit: The maximum number of samples to evaluate. Optional.
        seed: Random seed for reproducibility. Optional.
        tasks: A list of tasks (benchmarks) to evaluate (e.g., ["arc_challenge"]). Default is ["arc_challenge"].
        **extra_args: Any other arguments supported by lm_eval.evaluate.simple_evaluate (such as additional configuration options).


    Returns:
        A dictionary of the measured values.
    """
    # Call llm-eval-harness  to run the experiment

    # For vllm, you specify the vllm endpoint URL and the model name served by that endpoint.
    # The model name is passed as part of model_args, e.g. "vllm_api_url=...,model=<model_name>"
    # This does NOT use the OpenAI API; it uses your own vllm server.
    if not model_name:
        raise ValueError(
            "You must provide 'model_name' as an argument for vllm evaluation."
        )
    model_args = f"api_base={vllm_endpoint_url},model={model_name},tokenizer=gpt2"
    return simple_evaluate(
        model="openai-completions",
        model_args=model_args,
        batch_size=batch_size,
        tasks=tasks,
        num_fewshot=num_fewshot,
        limit=limit,
        random_seed=seed,
        **extra_args,
    )


def llm_eval_harness_hf_experiment(
    model: str,
    hf_token: str,
    batch_size: int,
    device: str = "cpu",
    num_fewshot: int = 0,
    limit: int | None = None,
    seed: int | None = None,
    tasks: list[str] = ["arc_challenge"],
    **extra_args,
) -> dict[str, typing.Any]:
    """
    Run an LLM eval harness experiment on a Hugging Face model.

    This function uses the lm_eval library to run the experiment.
    https://github.com/EleutherAI/lm-evaluation-harness

    Args:
        model: The model to use for the experiment (e.g., "meta-llama/Llama-3.1-8B-Instruct").
        hf_token: The HuggingFace token for model access.
        batch_size: The batch size to use for evaluation.
        device: The device to run the evaluation on (e.g., "cuda", "cpu"). Default is "cpu".
        num_fewshot: The number of few-shot examples to use. Default is 0.
        limit: The maximum number of samples to evaluate. Optional.
        seed: Random seed for reproducibility. Optional.
        tasks: A list of tasks (benchmarks) to evaluate (e.g., ["arc_challenge"]). Default is ["arc_challenge"].
        **extra_args: Any other arguments supported by lm_eval.evaluate.simple_evaluate (such as additional configuration options).

    Returns:
        A dictionary of the measured values.
    """
    # Call llm-eval-harness  to run the experiment

    model_args = f"pretrained={model},token={hf_token}"
    return simple_evaluate(
        model="hf",
        model_args=model_args,
        batch_size=batch_size,
        tasks=tasks,
        device=device,
        num_fewshot=num_fewshot,
        limit=limit,
        random_seed=seed,
        **extra_args,
    )


def llm_eval_harness_mmlu_experiment_vllm(
    vllm_endpoint_url: str,
    model_name: str,
    batch_size: int,
    device: str = "cpu",
    num_fewshot: int = 0,
    limit: int | None = None,
    seed: int | None = None,
    **extra_args,
) -> dict[str, typing.Any]:
    """
    Run the LLM eval harness MMLU experiment.

    This function uses the lm_eval library to run the experiment.
    https://github.com/EleutherAI/lm-evaluation-harness
    """
    results = llm_eval_harness_vllm_experiment(
        vllm_endpoint_url=vllm_endpoint_url,
        model_name=model_name,
        batch_size=batch_size,
        device=device,
        num_fewshot=num_fewshot,
        limit=limit,
        seed=seed,
        tasks=["mmlu"],
        **extra_args,
    )
    # Here we extract the critical metrics from the results
    accuracy = results["accuracy"]
    acc_norm = results["acc_norm"]
    return {"accuracy": accuracy, "acc_norm": acc_norm}


@ray.remote
def run_experiment(
    request: MeasurementRequest,
    experiment: Experiment | ParameterizedExperiment,
    measurement_queue: MeasurementQueue,
):

    # This function
    # 1. Performs the measurement represented by MeasurementRequest
    # 2. Updates MeasurementRequest with the results of the measurement and status
    # 3. Puts it in the measurement_queue

    measurements = []
    for entity in request.entities:

        #
        # Retrieve the input parameters to run your experiment on the entity
        #
        input_parameters = get_experiment_input_values(
            experiment=experiment, entity=entity
        )
        # You can implement the logic of your experiment inside the llm_eval_harness_experiment function above
        # Feel free to change the name
        #
        # NOTE: This simple case assumes there is only ONE experiment
        # If you want to support multiple experiments you would have to e.g. call different functions based on the passed experiment
        measured_values = llm_eval_harness_mmlu_experiment_vllm(**input_parameters)

        # Augment the values returned by llm_eval_harness_experiment to the structure used by ado
        measuredValues = [
            PropertyValue(
                value=identifier,
                property=experiment.observedPropertyForTargetIdentifier(identifier),
            )
            for identifier, value in measured_values.items()
        ]

        print(
            "Values for entity",
            entity.identifier,
            "and experiment",
            experiment.identifier,
            "experiment type is",
            type(experiment),
            "are",
            json.dumps(measured_values),
        )

        # Create a MeasurementResult to hold the results
        # This is used to
        # (a) Separate results from multiple entities
        # (b) Distinguish Valid and Invalid measurements -> especially in latter case to provide info on failure reasons

        # Here we use ValidMeasurementResult but if the experiment failed for some reason
        # We can use InvalidMeasurementResult and give a reason etc.
        measurements.append(
            ValidMeasurementResult(
                entityIdentifier=entity.identifier, measurements=measuredValues
            )
        )

    # For multi entity experiments if ONE entity had ValidResults the status must be SUCCESS
    request.status = MeasurementRequestStateEnum.SUCCESS
    request.measurements = measurements  # Note we don't set empty above as this would raise a validation error (can't have empty measurements)

    # Push the request to the state updates queue
    measurement_queue.put(request, block=False)
