package main

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"os"
	//"strconv"
)


// Users struct which contains
// an array of users
type Model struct {
	Models []RecvLayer `json:"Model"`
}

// User struct which contains a name
// a type and a list of social links
type RecvLayer struct {
	GScale float32 `json:"gscale"`
	SendLayer []RecvPrjn `json:"SendLayer"`
}

// Social struct which contains a
// list of links
type RecvPrjn struct {
	N int `json:"SendNeurons"`
	Si []int `json:"Synapse Index"`
	Wt []float32 `json:"Weights"`
}

func main() {
	// Open our jsonFile
	jsonFile, err := os.Open("SUMMER_Base_000_00032.wts")
	// if we os.Open returns an error then handle it
	if err != nil {
		fmt.Println(err)
	}

	fmt.Println("Successfully Opened weight files")
	// defer the closing of our jsonFile so that we can parse it later on
	defer jsonFile.Close()

	// read our opened xmlFile as a byte array.
	byteValue, _ := ioutil.ReadAll(jsonFile)

	// we iterate through every user within our users array and
	// print out the user Type, their name, and their facebook url
	// as just an example
	var result map[string]interface{}
	json.Unmarshal([]byte(byteValue), &result)

	fmt.Println(len(result))

}
