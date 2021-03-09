// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0

import React, { useEffect, useState } from "react";
import { Logo } from "../panel/logo/Logo";
import styles from "./Preloader.module.scss";
import classNames from "classnames";

interface Props {
  fullScreen?: boolean;
}

export const Preloader: React.FC<Props> = ({ fullScreen }) => {
  const [init, setInit] = useState(false);

  useEffect(() => {
    setTimeout(() => setInit(true), 1);
  }, []);

  return (
    <div
      className={classNames(
        styles.preloader,
        fullScreen && styles["preloader--full-screen"],
        init && styles["preloader--init"]
      )}
    >
      <Logo className={styles.logo} />
    </div>
  );
};
