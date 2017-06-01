open System
open System.IO
open Encog.ML.Data.Basic
open Encog.Engine.Network.Activation
open Encog.Neural.Networks
open Encog.Neural.Networks.Layers
open Encog.Neural.Networks.Training.Propagation.Resilient


let trainPath = "../../data/train.csv"
let testPath = "../../data/test.csv"
let resultsPath = "../../data/results.csv"


// gets a comma separated string and returns an Array of floats
let parse (line: String) =
    line.Split [|','|] |> Array.map float

// Transforms integer to an array that represents the ideal output from the network 
let toNetworkOutput (value: float) =
    let output = Array.create 10 0.0
    Array.set output (int value) 1.0
    output

// maps an array of 0-max range numbers to 0-1 floats 
let toZeroeOneRange max =
    Array.map (fun n -> n / max)

// Creates a network from an array that represents the number of neurons on each layer
let createNetwork (layers: List<int>) =
    let network = BasicNetwork()
    network.AddLayer( BasicLayer( null, true, (List.head layers) ))
    let rec addLayers layerList =
        let addLayer n =
            network.AddLayer( BasicLayer( ActivationElliott(), true, (List.head n) )) // try differnt activations: softmax sigmoid Elliot ...
            addLayers (List.tail n)
        match layerList with
        | x :: [] -> network.AddLayer( BasicLayer( ActivationSoftMax(), false, x ))
        | x :: xs -> addLayer (x :: xs)
        | _ -> () //should not happen
    List.tail layers |> addLayers 
    network.Structure.FinalizeStructure()
    network.Reset()
    network

// Transforms the encog output array into an integer prediction
let stringToOutput s =
    let toArr (s: String) =
        let rep1 = s.Replace("[", "")
        rep1.Replace("]", "") |> parse
    let toNumber arr =
        let tup = Array.fold (fun (index, maxIndex, maxValue) v -> 
                                match (v > maxValue) with
                                | true -> (index+1, index, v)
                                | false -> (index+1, maxIndex, maxValue)
                                ) (0, 0, 0.0) arr
        match tup with
        | (a, b, c) -> b
    (toArr >> toNumber) s

// Train a network
let train trainingSet (network: BasicNetwork) =
    let trainedNetwork = network.Clone() :?> BasicNetwork
    let trainer = ResilientPropagation(trainedNetwork, trainingSet)
    
    let rec trainIteration epoch (error: float) =
        match epoch <= 100 with
        | false -> ()
        | true -> trainer.Iteration()
                  printfn "Iteration no : %d, Error: %f" epoch error
                  trainIteration (epoch + 1) trainer.Error
    
    trainIteration 1 1.0
    trainedNetwork

// Trains the network and validates
let trainMNISTNetwork (network: BasicNetwork) =
    printfn "START READING"
    let parsedLines = File.ReadLines(trainPath) |> Seq.tail |> Seq.toArray |> Array.map parse
    printfn "START PROCESSING"
    let inputs = parsedLines |> Array.map Array.tail |> Array.map (toZeroeOneRange 255.0)
    let labels = parsedLines |> Array.map Array.head
    let idealOutputs = labels |> Array.map toNetworkOutput
    //total: 42000
    let trainSize = 10000
    let validationSize = 42000-trainSize
    //when using the entire test set the validation set will overlap, so it may be subject to overfitting
    let trainingSet = BasicMLDataSet(inputs, idealOutputs)
    //let trainingSet = BasicMLDataSet(Array.take trainSize inputs, Array.take trainSize idealOutputs)
    let validationSet = BasicMLDataSet((Array.skip trainSize inputs |> Array.take validationSize), (Array.skip trainSize idealOutputs |> Array.take validationSize))

    printfn "START TRAINING"
    let trainedNetwork = network |> train trainingSet

    let mutable correct = 0
    validationSet
    |> Seq.iter (
        fun item ->
            let output = trainedNetwork.Compute(item.Input)
            match ( (stringToOutput (item.Ideal.ToString())) = (stringToOutput (output.ToString())) ) with
            | true -> correct <- correct + 1
            | _ ->  correct <- correct)
    printfn "correct: %i, percentage: %f" correct ((float correct)/(float validationSize))
    
    trainedNetwork

// Outputs the test set predictions in a file for kaggle    
let outputMNISTNetwork (network: BasicNetwork) =
    let testInputs = File.ReadLines(testPath) |> Seq.tail |> Seq.toArray |> Array.map parse |> Array.map (toZeroeOneRange 255.0)
    let testingSet = BasicMLDataSet(testInputs, null)
    use streamWriter = new StreamWriter(resultsPath)
    streamWriter.WriteLine("ImageId,Label")
    
    let mutable index = 1
    testingSet
    |> Seq.iter (
        fun item ->
            let output = network.Compute(item.Input)
            let line = sprintf "%i,%i" index (stringToOutput (output.ToString()))
            streamWriter.WriteLine(line)
            index <- index + 1
            )


[<EntryPoint>]
let main argv = 
    let network = createNetwork [784; 100; 10]
    printfn "network: %A" network
    let trainedNetwork = trainMNISTNetwork network
    printfn "START TESTING"
    outputMNISTNetwork trainedNetwork
    printfn "DONE"
    0
