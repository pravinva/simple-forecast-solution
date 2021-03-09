// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0

import Amplify from "@aws-amplify/core";
import amplifyExports from "../amplify-out/aws-exports";

export const configureAmplify = () => {
  Amplify.configure(amplifyExports);
};
