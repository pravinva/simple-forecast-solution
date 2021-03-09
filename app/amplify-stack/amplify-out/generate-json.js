const fs = require("fs");
const vm = require("vm");

// Get aws exports code and ensure is ES5 compatible
let awsExportsCode = fs.readFileSync(__dirname + "/aws-exports.js").toString('utf8');
awsExportsCode = awsExportsCode.replace("export default", "module.exports = ");

const amplifyConfig = eval(awsExportsCode)

fs.writeFileSync(
  __dirname + "/aws-exports.json",
  JSON.stringify(amplifyConfig, null, 2)
);
