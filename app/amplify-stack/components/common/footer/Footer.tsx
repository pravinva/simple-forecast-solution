// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0

import React from "react";
import styles from "./Footer.module.scss";
import { Container } from "@material-ui/core";

export const Footer = () => {
  return (
    <footer className={styles.footer}>
      <Container className="container">
        <div className="copyright">&copy; 2020, Amazon Web Services, Inc. or its affiliates. All rights reserved.</div>
      </Container>
    </footer>
  );
};
