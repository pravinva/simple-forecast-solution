// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0

import React, { HTMLProps } from "react";
import style from "./List.module.scss";
import classNames from "classnames";

export const List: React.FC<HTMLProps<HTMLUListElement>> = ({ children, ...props }) => {
  return (
    <ul {...props} className={classNames(style.list, props.className)}>
      {children}
    </ul>
  );
};
