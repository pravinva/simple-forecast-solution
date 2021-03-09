// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0

import React from "react";
import styles from "./Button.module.scss";
import { ButtonProps, Button as MaterialButton } from "@material-ui/core";

export const Button: React.FC<ButtonProps> = ({ children, ...props }) => {
  return (
    <MaterialButton className={styles.button} {...props}>
      {children}
    </MaterialButton>
  );
};
